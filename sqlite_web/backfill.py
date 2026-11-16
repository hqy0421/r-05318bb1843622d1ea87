"""Recipe-driven code cleaning and dictionary backfill.

A recipe is an ordered list of normalization steps: trim, case folding,
regular-expression replacement and NULL filling.  The *same* recipe is
applied to the code column of a business table and to the code column of a
reference dictionary, so the two sides match on equal, cleaned codes.  Each
business code must resolve to exactly one dictionary row; no match or more
than one match is an error.

A preview reads the source rows, normalizes their codes, looks each code up
in the dictionary and works out what SQLite would actually store in the
business columns.  The "what SQLite stores" answer is obtained by inserting
the planned values into throw-away TEMP tables whose columns have the same
declared type (and STRICT-ness) as the real columns.  TEMP tables live in the
connection's temp database: they never touch the business table, and they
work on a read-only connection too.

Applying re-runs the planning step inside a single write transaction, checks
that the source data, dictionary, recipe and schema are still exactly the
ones that were previewed (a fingerprint), and then updates every changed row
by primary key.  Only the selected columns are written; the primary key and
every other column keep their value.  Any conversion or constraint error
(including CHECK, UNIQUE and foreign-key failures) aborts the transaction,
so the whole batch rolls back.

The code column itself is only rewritten by the recipe; the dictionary is
matched on the normalized code, not on its raw text.  SQL NULL, the text
"NULL" and BLOB values are distinct throughout: a NULL code is normalized
only by a "null" recipe step, the text "NULL" is ordinary text, and a BLOB
code is reported as an error instead of being coerced.
"""

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field

MAX_STEPS = 32
MAX_ROWS = 100000
PREVIEW_ROWS = 500
MAX_PREVIEW_ERRORS = 200
MAX_PATTERN = 500
MAX_REPLACEMENT = 500
MAX_FILL = 500
MAX_RECIPE_TEXT = 200000
MAX_MAPPINGS = 64
MAX_IDENTIFIER = 200

# Flags accepted by a "replace" step, mapped to their re module constants.
FLAG_MAP = {'i': re.I, 'm': re.M, 's': re.S, 'x': re.X}

DEFAULT_RECIPE_TEXT = '''[
  {"op": "trim"},
  {"op": "case", "mode": "upper"},
  {"op": "replace", "pattern": "\\\\s+", "replacement": ""},
  {"op": "null", "fill": "UNKNOWN"}
]'''

PROBE_TABLE = '_sqlite_web_backfill_probe'

SQLITE_KINDS = ('null', 'integer', 'real', 'text', 'blob')

MISSING = object()


class CleanError(Exception):
    """A configuration or data problem that must be shown to the user."""


class StalePreview(CleanError):
    def __init__(self, message, plan=None):
        super().__init__(message)
        self.plan = plan


class PlanRefused(CleanError):
    def __init__(self, message, plan=None):
        super().__init__(message)
        self.plan = plan


#
# Recipes.
#

class Recipe(object):
    def __init__(self, steps, patterns):
        self.steps = steps
        self._patterns = patterns
        self.json = json.dumps(steps, sort_keys=True, ensure_ascii=False,
                               separators=(',', ':'))

    def __repr__(self):
        return 'Recipe(%s)' % self.json


def _check_text(value, what, limit):
    if not isinstance(value, str):
        raise CleanError('%s must be a string.' % what)
    if len(value) > limit:
        raise CleanError('%s is longer than %d characters.' % (what, limit))
    return value


def _step_keys(step, index, required, optional=()):
    if not isinstance(step, dict):
        raise CleanError('Recipe step %d must be a JSON object.' % (index + 1))
    unknown = sorted(set(step) - set(required) - set(optional))
    if unknown:
        raise CleanError('Recipe step %d has unknown key(s): %s.' % (
            index + 1, ', '.join(unknown)))
    missing = [key for key in required if key not in step]
    if missing:
        raise CleanError('Recipe step %d is missing key(s): %s.' % (
            index + 1, ', '.join(missing)))


def parse_recipe(text):
    """Parse and validate a recipe.  Returns a Recipe.

    Invalid JSON, unknown steps, invalid regular expressions and oversized
    values are all rejected up-front, before any data is read.
    """
    if not isinstance(text, str):
        raise CleanError('Recipe must be JSON text.')
    if len(text) > MAX_RECIPE_TEXT:
        raise CleanError('Recipe is too large.')
    if not text.strip():
        raise CleanError('Recipe is required.')
    try:
        raw = json.loads(text)
    except ValueError as exc:
        raise CleanError('Recipe is not valid JSON: %s' % exc)
    if not isinstance(raw, list):
        raise CleanError('Recipe must be a JSON list of steps.')
    if len(raw) > MAX_STEPS:
        raise CleanError('Recipe has more than %d steps.' % MAX_STEPS)

    steps, patterns = [], []
    for index, step in enumerate(raw):
        if not isinstance(step, dict) or 'op' not in step:
            raise CleanError('Recipe step %d must be an object with an "op".'
                             % (index + 1))
        op = step['op']
        if op == 'trim':
            _step_keys(step, index, ('op',), ('chars',))
            canonical = {'op': 'trim'}
            if 'chars' in step:
                canonical['chars'] = _check_text(
                    step['chars'], 'trim chars', MAX_FILL)
            steps.append(canonical)
            patterns.append(None)
        elif op == 'case':
            _step_keys(step, index, ('op', 'mode'))
            mode = step['mode']
            if mode not in ('upper', 'lower'):
                raise CleanError('Recipe step %d: case mode must be "upper" '
                                 'or "lower".' % (index + 1))
            steps.append({'op': 'case', 'mode': mode})
            patterns.append(None)
        elif op == 'replace':
            _step_keys(step, index, ('op', 'pattern', 'replacement'),
                       ('flags',))
            pattern = _check_text(step['pattern'], 'replace pattern',
                                  MAX_PATTERN)
            if not pattern:
                raise CleanError('Recipe step %d: replace pattern must not be '
                                 'empty.' % (index + 1))
            replacement = _check_text(step['replacement'],
                                      'replace replacement', MAX_REPLACEMENT)
            flags_text = _check_text(
                step.get('flags', ''), 'replace flags', len(FLAG_MAP))
            # Flags are case-insensitive and canonicalized, so the same
            # recipe always produces the same fingerprint.
            flags_text = ''.join(sorted(set(flags_text.lower())))
            bad = sorted(set(flags_text) - set(FLAG_MAP))
            if bad:
                raise CleanError('Recipe step %d: unsupported regex flag(s): '
                                 '%s.' % (index + 1, ', '.join(bad)))
            flags_text = ''.join(sorted(set(flags_text)))
            flags = 0
            for ch in flags_text:
                flags |= FLAG_MAP[ch]
            try:
                compiled = re.compile(pattern, flags)
            except re.error as exc:
                raise CleanError('Recipe step %d: invalid regular expression: '
                                 '%s' % (index + 1, exc))
            canonical = {'op': 'replace', 'pattern': pattern,
                         'replacement': replacement}
            if flags_text:
                canonical['flags'] = flags_text
            steps.append(canonical)
            patterns.append(compiled)
        elif op == 'null':
            _step_keys(step, index, ('op', 'fill'), ('when',))
            fill = _check_text(step['fill'], 'null fill value', MAX_FILL)
            when = step.get('when', 'blank')
            if when not in ('blank', 'empty', 'null'):
                raise CleanError('Recipe step %d: null "when" must be '
                                 '"blank", "empty" or "null".' % (index + 1))
            canonical = {'op': 'null', 'fill': fill}
            if when != 'blank':
                canonical['when'] = when
            steps.append(canonical)
            patterns.append(None)
        else:
            raise CleanError('Recipe step %d: unknown op "%s". Supported ops '
                             'are trim, case, replace and null.'
                             % (index + 1, op))
    return Recipe(steps, patterns)


def apply_recipe(recipe, value):
    """Normalize one value.  Returns (value, error).

    None is SQL NULL: trim/case/replace leave it alone and only a null step
    can fill it.  The literal text "NULL" is ordinary text and is never
    treated as SQL NULL.  A BLOB cannot be normalized by a text recipe.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None, ('BLOB code cannot be normalized by a text recipe; '
                      'convert it to text first.')
    if value is None:
        current = None
    elif isinstance(value, bool):
        current = '1' if value else '0'
    elif isinstance(value, str):
        current = value
    else:
        current = str(value)

    for step, pattern in zip(recipe.steps, recipe._patterns):
        op = step['op']
        if op == 'null':
            when = step.get('when', 'blank')
            if current is None:
                current = step['fill']
            elif when == 'null':
                pass
            elif when == 'empty':
                if current == '':
                    current = step['fill']
            elif current.strip() == '':
                current = step['fill']
            continue
        if current is None:
            # trim/case/replace of NULL is NULL.
            continue
        if op == 'trim':
            chars = step.get('chars')
            current = current.strip(chars) if chars else current.strip()
        elif op == 'case':
            current = (current.upper() if step['mode'] == 'upper'
                       else current.lower())
        elif op == 'replace':
            current = pattern.sub(step['replacement'], current)
    return current, None


def parse_mapping(text):
    """Parse dictionary-column -> business-column mappings.

    Accepts one mapping per line ("code = code", "category -> category") or a
    JSON object.  Returns a list of (source, target) tuples.
    """
    if text is None:
        text = ''
    if not isinstance(text, str):
        raise CleanError('Column mapping must be text.')
    if text.strip().startswith('{'):
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise CleanError('Column mapping is not valid JSON: %s' % exc)
        if not isinstance(data, dict):
            raise CleanError('Column mapping must be a JSON object or one '
                             '"source = target" pair per line.')
        pairs = list(data.items())
    else:
        pairs = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            match = re.match(r'^(.+?)\s*(?:->|=)\s*(.+)$', line)
            if match is None:
                raise CleanError('Cannot parse mapping line "%s". Use '
                                 '"source = target".' % line)
            pairs.append((match.group(1).strip(), match.group(2).strip()))

    if not pairs:
        # Cleaning the code column alone is allowed; build_plan refuses an
        # empty mapping when no column would be updated.
        return []
    if len(pairs) > MAX_MAPPINGS:
        raise CleanError('Too many column mappings (maximum %d).'
                         % MAX_MAPPINGS)
    seen_targets = set()
    result = []
    for source, target in pairs:
        source = _check_text(source, 'mapping source column', MAX_IDENTIFIER)
        target = _check_text(target, 'mapping target column', MAX_IDENTIFIER)
        if not source or not target:
            raise CleanError('Mapping columns must not be empty.')
        if target in seen_targets:
            raise CleanError('Column "%s" is mapped more than once.' % target)
        seen_targets.add(target)
        result.append((source, target))
    return result


#
# Cells.
#

@dataclass(frozen=True)
class Cell(object):
    """A value together with the storage class SQLite uses for it."""
    kind: str
    value: object = None

    @property
    def is_null(self):
        return self.kind == 'null'


def cell_from_value(value):
    if value is None:
        return Cell('null', None)
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return Cell('blob', bytes(value))
    if isinstance(value, bool):
        return Cell('integer', int(value))
    if isinstance(value, int):
        return Cell('integer', value)
    if isinstance(value, float):
        return Cell('real', value)
    return Cell('text', str(value))


def _cell_from_typeof(kind, value):
    if kind not in SQLITE_KINDS:
        kind = 'text'
    if kind == 'null':
        return Cell('null', None)
    if kind == 'blob':
        return Cell('blob', bytes(value))
    if kind == 'integer':
        return Cell('integer', int(value))
    if kind == 'real':
        return Cell('real', float(value))
    return Cell('text', value)


def _same_cell(left, right):
    if left is None or right is None:
        return left is right
    return left.kind == right.kind and left.value == right.value


#
# Table metadata.
#

@dataclass
class Column(object):
    name: str
    type: str
    notnull: bool
    default: object
    pk: int
    hidden: int

    @property
    def writable(self):
        # hidden 0 = ordinary column.  Generated (2, 3) and virtual-table
        # hidden (1) columns cannot be written.
        return self.hidden == 0

    @property
    def readable(self):
        # Hidden columns of virtual tables (hidden == 1) are not part of
        # SELECT *.  Generated columns are readable by name.
        return self.hidden in (0, 2, 3)


@dataclass
class TableInfo(object):
    name: str
    type: str
    without_rowid: bool
    strict: bool
    columns: list

    def column(self, name):
        for column in self.columns:
            if column.name == name:
                return column
        return None

    @property
    def pk_columns(self):
        return sorted((c for c in self.columns if c.pk), key=lambda c: c.pk)

    @property
    def readable_columns(self):
        return [c for c in self.columns if c.readable]


def _quote_ident(name):
    return '"%s"' % name.replace('"', '""')


def _quote_type(decl_type):
    return '"%s"' % decl_type.replace('"', '""')


def _table_flags(dataset, name):
    """Return (type, without_rowid, strict) for a table in main."""
    try:
        row = dataset.query(
            'SELECT type, wr, strict FROM pragma_table_list '
            "WHERE schema = 'main' AND name = ?", (name,)).fetchone()
    except Exception:
        row = None
    if row is not None:
        return row[0], bool(row[1]), bool(row[2])

    # Fallback for SQLite before 3.37 (no pragma_table_list): parse the
    # stored CREATE statement.
    row = dataset.query(
        "SELECT type, sql FROM sqlite_master WHERE name = ? AND "
        "type IN ('table', 'view')", (name,)).fetchone()
    if row is None:
        return None, False, False
    table_type, sql = row
    tail = ''
    match = re.search(r'\)\s*(.*)$', sql or '', re.S)
    if match:
        tail = re.sub(r'--[^\n]*', ' ', match.group(1))
    upper = ' '.join(tail.upper().split())
    without_rowid = 'WITHOUT ROWID' in upper
    strict = re.search(r'\bSTRICT\b', upper) is not None
    return table_type, without_rowid, strict


def _table_info(dataset, name, label):
    if not name:
        raise CleanError('%s name is required.' % label)
    table_type, without_rowid, strict = _table_flags(dataset, name)
    if table_type is None:
        raise CleanError('%s "%s" was not found.' % (label, name))
    if table_type != 'table':
        raise CleanError('%s "%s" is a %s; only ordinary tables can be '
                         'cleaned.' % (label, name, table_type))
    rows = dataset.query(
        'SELECT cid, name, type, "notnull", dflt_value, pk, hidden '
        'FROM pragma_table_xinfo(?)', (name,)).fetchall()
    columns = [Column(row[1], row[2] or '', bool(row[3]), row[4], int(row[5]),
                      int(row[6])) for row in rows]
    return TableInfo(name, table_type, without_rowid, strict, columns)


def _order_by(info):
    # Order by the primary key, which is unique and always present once
    # build_plan has checked for one.  It is stable even if a user column is
    # named "rowid" (which would shadow the implicit rowid) and also covers
    # WITHOUT ROWID tables.
    cols = info.pk_columns
    if cols:
        return ', '.join(_quote_ident(c.name) for c in cols)
    if info.without_rowid:
        raise CleanError('Table "%s" has no primary key.' % info.name)
    return 'rowid'


def _read_rows(dataset, info):
    columns = info.readable_columns
    sql = 'SELECT %s FROM main.%s ORDER BY %s' % (
        ', '.join(_quote_ident(c.name) for c in columns),
        _quote_ident(info.name),
        _order_by(info))
    return columns, dataset.query(sql).fetchall()


#
# Fingerprint.
#

def _canonical(value):
    if value is None:
        return None
    if isinstance(value, memoryview):
        value = bytes(value)
    if isinstance(value, (bytes, bytearray)):
        return {'b': base64.b64encode(bytes(value)).decode('ascii')}
    if isinstance(value, bool):
        return {'i': int(value)}
    if isinstance(value, int):
        return {'i': value}
    if isinstance(value, float):
        if value != value:
            return {'f': 'nan'}
        if value == float('inf'):
            return {'f': 'inf'}
        if value == float('-inf'):
            return {'f': '-inf'}
        return {'f': repr(value)}
    return {'s': str(value)}


def _fingerprint(dataset, config, binfo, dinfo, bcolumns, brows, dcolumns,
                 drows):
    digest = hashlib.sha256()

    def feed(tag, obj):
        digest.update(tag.encode('utf8'))
        digest.update(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                 separators=(',', ':')).encode('utf8'))
        digest.update(b'\n')

    feed('config', {
        'table': config.table,
        'code_column': config.code_column,
        'update_code': bool(config.update_code),
        'dict_table': config.dict_table,
        'dict_code_column': config.dict_code_column,
        'mapping': [list(pair) for pair in config.mapping],
        'recipe': config.recipe.json,
    })
    for info in (binfo, dinfo):
        feed('columns', {
            'name': info.name,
            'type': info.type,
            'without_rowid': info.without_rowid,
            'strict': info.strict,
            'columns': [[c.name, c.type, c.notnull, _canonical(c.default),
                         c.pk, c.hidden] for c in info.columns],
        })
    schema_rows = dataset.query(
        'SELECT type, name, tbl_name, sql FROM sqlite_master '
        'WHERE tbl_name IN (?, ?) ORDER BY type, name',
        (config.table, config.dict_table)).fetchall()
    feed('schema', [list(row) for row in schema_rows])
    feed('source_columns', [c.name for c in bcolumns])
    feed('source_rows', len(brows))
    for row in brows:
        feed('row', [_canonical(value) for value in row])
    feed('dict_columns', [c.name for c in dcolumns])
    feed('dict_rows', len(drows))
    for row in drows:
        feed('drow', [_canonical(value) for value in row])
    return digest.hexdigest()


#
# Plan.
#

@dataclass
class RowPlan(object):
    index: int
    pk: tuple
    pk_label: str
    code_before: Cell = None
    code_after: Cell = None
    norm_code: str = None
    match_state: str = 'bad'  # ok | none | many | bad
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    writes: dict = field(default_factory=dict)
    changes: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)

    @property
    def changed(self):
        return bool(self.changes)

    @property
    def error(self):
        return '; '.join(self.errors)


@dataclass
class Plan(object):
    config: object
    fingerprint: str = ''
    rows: list = field(default_factory=list)
    display_rows: list = field(default_factory=list)
    error_rows: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    updated_columns: list = field(default_factory=list)
    pk_columns: list = field(default_factory=list)
    mapping: list = field(default_factory=list)
    truncated: bool = False
    applied: int = 0

    @property
    def error_count(self):
        return self.counts.get('error_rows', 0)


def _pk_label(values):
    parts = []
    for value in values:
        cell = cell_from_value(value)
        if cell.kind == 'blob':
            parts.append("X'%s'" % cell.value.hex())
        elif cell.kind == 'null':
            parts.append('NULL')
        elif cell.kind == 'text':
            parts.append("'%s'" % cell.value)
        else:
            parts.append(str(cell.value))
    return ', '.join(parts)


def _quoted(value):
    return "'%s'" % str(value).replace("'", "''")


def _probe_stored_values(dataset, binfo, column, values, errors_by_row):
    """Work out what SQLite stores for `values` in `column`.

    `values` holds one entry per source row; MISSING means the row has no
    planned value for this column.  Returns {row_index: Cell}.  Insertion
    errors (STRICT type mismatches and so on) are recorded per row.

    TEMP tables live in the connection's temp database, so they work on a
    read-only connection and never touch the business table.
    """
    probe = PROBE_TABLE
    qprobe = _quote_ident(probe)
    try:
        dataset.query('DROP TABLE IF EXISTS temp.%s' % qprobe)
        if binfo.strict and column.type:
            dataset.query('CREATE TEMP TABLE %s (v %s) STRICT' % (
                qprobe, _quote_type(column.type)))
        else:
            # A CTAS from the real column preserves its affinity exactly.
            # PRAGMA table_xinfo's type text cannot always do that (a
            # quoted-empty declaration and one we re-quote differ, and a
            # CTAS also spells NUMERIC-affinity types as "NUM").
            dataset.query(
                'CREATE TEMP TABLE %s AS SELECT %s AS v FROM main.%s '
                'WHERE 0' % (qprobe, _quote_ident(column.name),
                             _quote_ident(binfo.name)))
    except Exception as exc:
        raise CleanError('Could not inspect how SQLite would store column '
                         '"%s": %s' % (column.name, exc))

    pairs = [(index, value) for index, value in enumerate(values)
             if value is not MISSING]
    result = {}
    try:
        if pairs:
            sql = 'INSERT INTO temp.%s(rowid, v) VALUES (?, ?)' % qprobe
            conn = dataset._database.connection()
            try:
                conn.executemany(sql, [(index + 1, value)
                                       for index, value in pairs])
            except Exception:
                # executemany stops at the first failing row without saying
                # which one.  Retry one row at a time so each error is
                # attributed to its source row.
                dataset.query('DELETE FROM temp.%s' % qprobe)
                for index, value in pairs:
                    try:
                        dataset.query(sql, (index + 1, value))
                    except Exception as exc:
                        message = str(exc).replace('%s.v' % probe,
                                                   '"%s"' % column.name)
                        errors_by_row.setdefault(index, []).append(
                            'Column "%s": %s' % (column.name, message))
        rows = dataset.query(
            'SELECT rowid, typeof(v), v FROM temp.%s' % qprobe).fetchall()
        for rowid, kind, value in rows:
            result[int(rowid) - 1] = _cell_from_typeof(kind, value)
    finally:
        dataset.query('DROP TABLE IF EXISTS temp.%s' % qprobe)
    return result


@dataclass
class BackfillConfig(object):
    table: str
    code_column: str
    dict_table: str
    dict_code_column: str
    mapping: list
    recipe: object
    update_code: bool = True

    @property
    def mapping_text(self):
        return '\n'.join('%s = %s' % pair for pair in self.mapping)

    @property
    def recipe_text(self):
        return self.recipe.json


def parse_config(table, code_column, dict_table, dict_code_column,
                  mapping_text, recipe_text, update_code):
    if not code_column:
        raise CleanError('Select the business table code column.')
    if not dict_table:
        raise CleanError('Enter the reference dictionary table.')
    if not dict_code_column:
        raise CleanError('Enter the dictionary code column.')
    recipe = parse_recipe(recipe_text)
    mapping = parse_mapping(mapping_text)
    return BackfillConfig(
        table=table,
        code_column=code_column,
        dict_table=dict_table.strip(),
        dict_code_column=dict_code_column.strip(),
        mapping=mapping,
        recipe=recipe,
        update_code=bool(update_code))


def build_plan(dataset, config):
    """Read the data and build a full preview plan.

    Never writes to the business table or the dictionary.
    """
    binfo = _table_info(dataset, config.table, 'Table')
    dinfo = _table_info(dataset, config.dict_table, 'Dictionary table')

    code_column = binfo.column(config.code_column)
    if code_column is None or not code_column.readable:
        raise CleanError('Column "%s" is not a readable column of table "%s".'
                         % (config.code_column, config.table))
    if config.update_code and not code_column.writable:
        raise CleanError('Column "%s" cannot be written, so the code column '
                         'cannot be rewritten.' % config.code_column)
    dict_code_column = dinfo.column(config.dict_code_column)
    if dict_code_column is None or not dict_code_column.readable:
        raise CleanError('Column "%s" was not found in dictionary table "%s".'
                         % (config.dict_code_column, config.dict_table))
    if config.dict_table == config.table:
        raise CleanError('The dictionary must be a different table: cleaning '
                         '"%s" against itself would change the reference it '
                         'is matched on.' % config.table)

    pk_columns = binfo.pk_columns
    if not pk_columns:
        raise CleanError('Table "%s" has no primary key. Cleaning updates '
                         'rows by primary key, so one is required.'
                         % config.table)
    pk_names = [c.name for c in pk_columns]
    if config.update_code and config.code_column in pk_names:
        raise CleanError('The code column "%s" is part of the primary key, '
                         'so rewriting it would change the primary key. '
                         'Uncheck "rewrite the business code column" or pick '
                         'a different code column.' % config.code_column)

    updated_columns = []
    if config.update_code:
        updated_columns.append(config.code_column)
    for source, target in config.mapping:
        dsource = dinfo.column(source)
        if dsource is None or not dsource.readable:
            raise CleanError('Mapping source column "%s" was not found in '
                             'dictionary table "%s".' % (source,
                                                         config.dict_table))
        tcolumn = binfo.column(target)
        if tcolumn is None or not tcolumn.writable:
            raise CleanError('Mapping target column "%s" is not a writable '
                             'column of table "%s".' % (target,
                                                        config.table))
        if target in pk_names:
            raise CleanError('Cannot backfill primary key column "%s"; the '
                             'primary key must stay unchanged.' % target)
        if config.update_code and target == config.code_column:
            raise CleanError('The code column "%s" cannot also be a mapping '
                             'target.' % target)
        if target not in updated_columns:
            updated_columns.append(target)
    if not updated_columns:
        raise CleanError('Nothing would be updated: enable "rewrite the code '
                         'column" or map at least one dictionary column.')

    bcolumns, brows = _read_rows(dataset, binfo)
    dcolumns, drows = _read_rows(dataset, dinfo)
    if len(brows) > MAX_ROWS:
        raise CleanError('Table "%s" has more than %d source rows; use the '
                         'Query tab to build a smaller working table first.'
                         % (config.table, MAX_ROWS))
    if len(drows) > MAX_ROWS:
        raise CleanError('Dictionary "%s" has more than %d rows.'
                         % (config.dict_table, MAX_ROWS))

    bpos = {c.name: i for i, c in enumerate(bcolumns)}
    dpos = {c.name: i for i, c in enumerate(dcolumns)}

    # Normalize dictionary codes once.  Unusable dictionary rows are
    # skipped with a warning; a business row that needs one then simply
    # reports "no match".
    warnings = []
    entries = []
    ignored = 0
    for index, row in enumerate(drows):
        raw = row[dpos[dict_code_column.name]]
        norm, error = apply_recipe(config.recipe, raw)
        reason = error
        if reason is None and (norm is None or norm == ''):
            reason = 'its code is empty after applying the recipe'
        if reason is not None:
            ignored += 1
            if len(warnings) < MAX_PREVIEW_ERRORS:
                warnings.append('Dictionary row %d was skipped: %s.'
                                % (index + 1, reason))
            continue
        values = {target: row[dpos[source]] for source, target in
                  config.mapping}
        entries.append((norm, values))

    by_code = {}
    for norm, values in entries:
        by_code.setdefault(norm, []).append(values)
    duplicate_codes = sum(1 for group in by_code.values() if len(group) > 1)

    rows = []
    for index, row in enumerate(brows):
        pk = tuple(row[bpos[name]] for name in pk_names)
        raw_code = row[bpos[config.code_column]]
        row_plan = RowPlan(
            index=index,
            pk=pk,
            pk_label=_pk_label(pk),
            code_before=cell_from_value(raw_code),
            before={target: cell_from_value(row[bpos[target]])
                    for _source, target in config.mapping})
        rows.append(row_plan)

        norm, error = apply_recipe(config.recipe, raw_code)
        if error is None and norm is None:
            error = ('The code is NULL and the recipe has no "null" step to '
                     'fill it.')
        elif error is None and norm == '':
            error = 'The code is empty after applying the recipe.'
        if error is not None:
            row_plan.errors.append(error)
            row_plan.match_state = 'bad'
            continue

        row_plan.norm_code = norm
        if config.update_code:
            row_plan.writes[config.code_column] = norm
        matches = by_code.get(norm, [])
        if not matches:
            row_plan.match_state = 'none'
            row_plan.errors.append(
                'No dictionary row matches code %s.' % _quoted(norm))
        elif len(matches) > 1:
            row_plan.match_state = 'many'
            row_plan.errors.append(
                '%d dictionary rows match code %s; exactly one match is '
                'required.' % (len(matches), _quoted(norm)))
        else:
            row_plan.match_state = 'ok'
            for _source, target in config.mapping:
                value = matches[0][target]
                row_plan.writes[target] = value
                if value is None and binfo.column(target).notnull:
                    row_plan.errors.append(
                        'Column "%s" does not allow NULL, but the dictionary '
                        'value is NULL.' % target)

    # Ask SQLite what it would store, using TEMP probe tables.  One probe
    # per updated column gives precise per-row error attribution.
    errors_by_row = {row.index: row.errors for row in rows}
    for column_name in updated_columns:
        column = binfo.column(column_name)
        values = [row.writes.get(column_name, MISSING) for row in rows]
        if not any(value is not MISSING for value in values):
            continue
        stored = _probe_stored_values(dataset, binfo, column, values,
                                      errors_by_row)
        for row in rows:
            cell = stored.get(row.index)
            if cell is None:
                continue
            if column_name == config.code_column and config.update_code:
                row.code_after = cell
            else:
                row.after[column_name] = cell

    changed = unchanged = 0
    for row in rows:
        code_changed = (config.update_code and row.code_after is not None and
                        not _same_cell(row.code_before, row.code_after))
        if code_changed:
            row.changes[config.code_column] = row.writes[config.code_column]
        for _source, target in config.mapping:
            after = row.after.get(target)
            if after is not None and not _same_cell(row.before.get(target),
                                                    after):
                row.changes[target] = row.writes[target]
        if row.errors:
            # Error rows are reported separately; they would never be
            # written because an error refuses the whole batch.
            row.changes.clear()
        elif row.changes:
            changed += 1
        else:
            unchanged += 1

    error_rows = [row for row in rows if row.errors]
    counts = {
        'source_rows': len(rows),
        'dictionary_rows': len(drows),
        'dictionary_used': len(entries),
        'dictionary_skipped': ignored,
        'dictionary_duplicate_codes': duplicate_codes,
        'matched_rows': sum(1 for row in rows if row.match_state == 'ok'),
        'unmatched_rows': sum(1 for row in rows if row.match_state == 'none'),
        'ambiguous_rows': sum(1 for row in rows if row.match_state == 'many'),
        'invalid_rows': sum(1 for row in rows if row.match_state == 'bad'),
        'error_rows': len(error_rows),
        'changed_rows': changed,
        'unchanged_rows': unchanged,
    }

    fingerprint = _fingerprint(dataset, config, binfo, dinfo, bcolumns,
                               brows, dcolumns, drows)
    return Plan(
        config=config,
        fingerprint=fingerprint,
        rows=rows,
        display_rows=rows[:PREVIEW_ROWS],
        error_rows=[row for row in error_rows[:MAX_PREVIEW_ERRORS]],
        warnings=warnings,
        counts=counts,
        updated_columns=updated_columns,
        pk_columns=pk_names,
        mapping=list(config.mapping),
        truncated=len(rows) > PREVIEW_ROWS)


def apply_plan(dataset, config, expected_fingerprint):
    """Re-plan and apply in one transaction; all-or-nothing.

    Returns the freshly built plan with .applied set to the number of
    updated rows.  Raises StalePreview / PlanRefused / CleanError; in every
    failure case no business row is modified.
    """
    if getattr(dataset, 'is_readonly', False):
        raise CleanError('The database is open read-only; changes cannot be '
                         'applied.')
    db = dataset._database
    conn = db.connection()
    if conn.in_transaction:
        raise CleanError('A transaction is already open; try again.')

    plan = None
    try:
        conn.execute('BEGIN IMMEDIATE')
        plan = build_plan(dataset, config)
        if not expected_fingerprint:
            raise StalePreview(
                'Preview the changes before applying them.', plan)
        if expected_fingerprint != plan.fingerprint:
            raise StalePreview(
                'The source data, dictionary, recipe or table structure '
                'changed since the preview. Review the new preview and '
                'confirm again.', plan)
        if plan.error_count:
            sample = plan.error_rows[0]
            raise PlanRefused(
                '%d row(s) have errors, so nothing was updated. First error '
                '(row %s): %s' % (plan.error_count, sample.pk_label,
                                  sample.error), plan)

        updated = 0
        for row in plan.rows:
            if not row.changes:
                continue
            columns = list(row.changes)
            sql = 'UPDATE main.%s SET %s WHERE %s' % (
                _quote_ident(config.table),
                ', '.join('%s = ?' % _quote_ident(name) for name in columns),
                ' AND '.join('%s IS ?' % _quote_ident(name)
                             for name in plan.pk_columns))
            params = [row.changes[name] for name in columns]
            params.extend(row.pk)
            try:
                cursor = dataset.query(sql, params)
            except Exception as exc:
                raise CleanError(
                    'Updating row %s failed: %s The whole batch was rolled '
                    'back.' % (row.pk_label, exc))
            if cursor.rowcount != 1:
                raise CleanError(
                    'Row %s matched %s rows instead of one; the batch was '
                    'rolled back.' % (row.pk_label, cursor.rowcount))
            updated += 1
        conn.commit()
        plan.applied = updated
        return plan
    except BaseException:
        try:
            conn.rollback()
        except Exception:
            # A trigger's RAISE(ROLLBACK) may already have rolled back.
            pass
        raise
