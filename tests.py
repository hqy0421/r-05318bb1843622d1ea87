import os
import shutil
import sqlite3
import tempfile
import unittest

from peewee import SqliteDatabase
from playhouse.dataset import DataSet

from sqlite_web import sqlite_web as sw
from sqlite_web import backfill as clean
from sqlite_web.executor import Result
from sqlite_web.executor import is_read
from sqlite_web.executor import key_decode
from sqlite_web.executor import key_encode
from sqlite_web.executor import run_one
from sqlite_web.executor import run_script
from sqlite_web.executor import split_statements
from sqlite_web.executor import wrap


class BaseExecutorTestCase(unittest.TestCase):
    def setUp(self):
        self.db = SqliteDatabase(':memory:')
        self.dataset = DataSet(self.db)
        self.dataset.query('CREATE TABLE users (id INTEGER PRIMARY KEY, '
                           'username TEXT)')
        for username in ('huey', 'mickey', 'zaizee'):
            self.dataset.query('INSERT INTO users (username) VALUES (?)',
                               (username,))

    def user_count(self):
        return self.dataset.query('SELECT COUNT(*) FROM users').fetchone()[0]


class TestRunOne(BaseExecutorTestCase):
    def test_read_paginates(self):
        r = run_one(self.dataset, 'SELECT * FROM users', page_size=2)
        self.assertEqual(r.kind, 'rows')
        self.assertEqual(r.columns, ['id', 'username'])
        self.assertEqual(len(r.rows), 2)
        self.assertTrue(r.has_next)
        self.assertIsNone(r.keys)

        r = run_one(self.dataset, 'SELECT * FROM users', page=2, page_size=2)
        self.assertEqual(len(r.rows), 1)
        self.assertFalse(r.has_next)

    def test_page_bounds(self):
        r = run_one(self.dataset, 'SELECT * FROM users', page=0, page_size=2)
        self.assertEqual(len(r.rows), 2)

        r = run_one(self.dataset, 'SELECT * FROM users', page=99, page_size=2)
        self.assertEqual(r.kind, 'rows')
        self.assertEqual(r.rows, [])
        self.assertFalse(r.has_next)

    def test_ordering(self):
        r = run_one(self.dataset, 'SELECT * FROM users', ordering=-2)
        self.assertEqual(r.rows[0][1], 'zaizee')
        r = run_one(self.dataset, 'SELECT * FROM users', ordering=2)
        self.assertEqual(r.rows[0][1], 'huey')

    def test_write_autocommits(self):
        r = run_one(self.dataset, "INSERT INTO users (username) VALUES ('x')")
        self.assertEqual(r.kind, 'affected')
        self.assertEqual(r.affected, 1)
        self.assertFalse(self.db.connection().in_transaction)
        self.assertEqual(self.user_count(), 4)

    def test_ddl(self):
        r = run_one(self.dataset, 'CREATE TABLE t2 (id INTEGER)')
        self.assertEqual(r.kind, 'affected')
        self.assertEqual(r.affected, -1)
        self.assertEqual(run_one(self.dataset, 'SELECT * FROM t2').kind,
                         'rows')

    def test_pragma(self):
        r = run_one(self.dataset, 'PRAGMA journal_mode')
        self.assertEqual(r.kind, 'rows')
        self.assertEqual(len(r.rows), 1)

    def test_returning(self):
        r = run_one(self.dataset,
                    "INSERT INTO users (username) VALUES ('r') RETURNING *")
        self.assertEqual(r.kind, 'rows')
        self.assertEqual(len(r.rows), 1)
        self.assertEqual(self.user_count(), 4)

    def test_error(self):
        r = run_one(self.dataset, 'SELECT nocolumn FROM users')
        self.assertEqual(r.kind, 'error')
        self.assertIn('nocolumn', r.error)

    def test_multi_statement_is_error(self):
        r = run_one(self.dataset, 'SELECT 1; SELECT 2')
        self.assertEqual(r.kind, 'error')

    def test_trailing_junk(self):
        for sql in ('SELECT * FROM users -- a comment',
                    'SELECT * FROM users;',
                    'SELECT * FROM users; \n ;'):
            r = run_one(self.dataset, sql)
            self.assertEqual(r.kind, 'rows', sql)
            self.assertEqual(len(r.rows), 3, sql)


class TestSplitStatements(unittest.TestCase):
    def test_split(self):
        script = ('CREATE TABLE t1 (id INTEGER);\n'
                  'CREATE TRIGGER trg AFTER INSERT ON t1 BEGIN '
                  'UPDATE t1 SET id = id; END;\n'
                  'INSERT INTO t1 VALUES (1);')
        stmts = split_statements(script)
        self.assertEqual(len(stmts), 3)
        self.assertTrue(stmts[1].startswith('CREATE TRIGGER'))
        self.assertTrue(stmts[1].endswith('END;'))

    def test_semicolon_in_string(self):
        self.assertEqual(split_statements("SELECT ';'; SELECT 2;"),
                         ["SELECT ';';", 'SELECT 2;'])

    def test_trailing_comment_chunk(self):
        self.assertEqual(split_statements('SELECT 1; -- done'),
                         ['SELECT 1;', '-- done'])


class TestRunScript(BaseExecutorTestCase):
    def run_sql(self, script, **kwargs):
        return run_script(self.dataset, split_statements(script), **kwargs)

    def test_statements_apply_independently(self):
        results = self.run_sql(
            "INSERT INTO users (username) VALUES ('a');"
            "INSERT INTO users (username) VALUES ('b');")
        self.assertEqual([r.kind for r in results], ['affected', 'affected'])
        self.assertFalse(self.db.connection().in_transaction)
        self.assertEqual(self.user_count(), 5)

    def test_stops_at_first_error(self):
        results = self.run_sql(
            "INSERT INTO users (username) VALUES ('a');"
            'CREATE TABLE t2 (id INTEGER);'
            'SELECT nocolumn FROM users;'
            "INSERT INTO users (username) VALUES ('never');")
        self.assertEqual([r.kind for r in results],
                         ['affected', 'affected', 'error'])
        # Statements before the error stay applied, DDL included.
        self.assertEqual(self.user_count(), 4)
        self.assertEqual(run_one(self.dataset, 'SELECT * FROM t2').kind,
                         'rows')

    def test_select_inside_script(self):
        results = self.run_sql(
            'SELECT * FROM users; SELECT COUNT(*) FROM users;', page_size=2)
        self.assertEqual(len(results[0].rows), 2)
        self.assertTrue(results[0].has_next)
        self.assertEqual(results[1].rows[0][0], 3)

    def test_user_owned_transaction(self):
        results = self.run_sql(
            "BEGIN; INSERT INTO users (username) VALUES ('a'); COMMIT;")
        self.assertEqual([r.kind for r in results], ['affected'] * 3)
        self.assertFalse(self.db.connection().in_transaction)
        self.assertEqual(self.user_count(), 4)

        self.run_sql(
            "BEGIN; INSERT INTO users (username) VALUES ('b'); ROLLBACK;")
        self.assertEqual(self.user_count(), 4)

    def test_dangling_begin_rolls_back_on_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = SqliteDatabase(os.path.join(tmpdir, 'x.db'))
            ds = DataSet(db)
            ds.query('CREATE TABLE t1 (id INTEGER)')
            results = run_script(ds, split_statements(
                'BEGIN; INSERT INTO t1 VALUES (1); SELECT nocolumn FROM t1;'))
            self.assertEqual(results[-1].kind, 'error')
            self.assertTrue(db.connection().in_transaction)
            # Request teardown closes the connection; sqlite rolls back.
            db.close()
            db.connect()
            self.assertEqual(
                ds.query('SELECT COUNT(*) FROM t1').fetchone()[0], 0)
            db.close()


class TestIsRead(BaseExecutorTestCase):
    def test_is_read(self):
        self.assertTrue(is_read(self.dataset, 'SELECT * FROM users'))
        self.assertTrue(is_read(self.dataset, 'SELECT * FROM users -- x'))
        self.assertFalse(is_read(self.dataset, 'DROP TABLE users'))
        self.assertFalse(is_read(self.dataset,
                                 "UPDATE users SET username = 'x'"))
        self.assertFalse(is_read(self.dataset, 'SELECT 1; SELECT 2'))
        self.assertEqual(self.user_count(), 3)


class TestWrap(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(wrap('SELECT 1;'),
                         'SELECT * FROM (\nSELECT 1\n) AS _')
        self.assertEqual(wrap('SELECT 1', ordering=-2),
                         'SELECT * FROM (\nSELECT 1\n) AS _ ORDER BY 2 DESC')
        self.assertEqual(wrap('SELECT 1', ordering=2, limit=51, offset=50),
                         'SELECT * FROM (\nSELECT 1\n) AS _ '
                         'ORDER BY 2 ASC LIMIT 51 OFFSET 50')
        self.assertEqual(wrap('SELECT 1', limit=0),
                         'SELECT * FROM (\nSELECT 1\n) AS _ LIMIT 0 OFFSET 0')
        self.assertEqual(wrap('SELECT 1', select='COUNT(*)'),
                         'SELECT COUNT(*) FROM (\nSELECT 1\n) AS _')


class TestRowKey(unittest.TestCase):
    def test_round_trips(self):
        for values in ([42], ['abc'], [b'\x01\x02\xff'], ['US', 'A:::B'],
                       [None, 1.5], ['✓']):
            self.assertEqual(key_decode(key_encode(values)), values)

    def test_url_safe(self):
        token = key_encode([b'\xfb\xff' * 30])
        self.assertNotIn('+', token)
        self.assertNotIn('/', token)


class BaseAppTestCase(unittest.TestCase):
    SCHEMA = """
        CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT);
        INSERT INTO users (username) VALUES ('huey'), ('mickey'), ('zaizee');
        CREATE TABLE comp (a TEXT, b TEXT, label TEXT, PRIMARY KEY (a, b));
        INSERT INTO comp VALUES ('US', 'A:::B', 'composite-row');
        CREATE TABLE blobs (id BLOB PRIMARY KEY, note TEXT);
        CREATE TABLE parent (id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO parent (name) VALUES ('p-one');
        CREATE TABLE child (id INTEGER PRIMARY KEY,
            parent_id INTEGER REFERENCES parent, label TEXT);
        INSERT INTO child (parent_id, label) VALUES (1, 'c-one');
        CREATE TABLE nopk (a TEXT);
        INSERT INTO nopk VALUES ('no-pk-row');
        CREATE TABLE oddpk ("user id" INTEGER NOT NULL, grp TEXT NOT NULL,
            val TEXT, PRIMARY KEY ("user id", grp));
        INSERT INTO oddpk VALUES (7, 'a', 'odd-row'), (8, 'a', 'same-grp');
        CREATE TABLE tag (name TEXT PRIMARY KEY);
        INSERT INTO tag VALUES (''), ('red');
        CREATE TABLE post (id INTEGER PRIMARY KEY,
            tag TEXT REFERENCES tag(name), body TEXT);
        CREATE VIEW v_users AS SELECT * FROM users;
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, 'app.db')
        conn = sqlite3.connect(self.db_path)
        conn.executescript(self.SCHEMA)
        conn.execute('INSERT INTO blobs VALUES (?, ?)', (b'\x00\xff', 'blob'))
        conn.commit()
        conn.close()
        sw.datasets.clear()
        sw.initialize_app([self.db_path])
        sw.app.config['TESTING'] = True
        self.client = sw.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def dbrows(self, sql, *params):
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return rows


class TestExecutionPolicy(BaseAppTestCase):
    def test_cross_site_post_rejected(self):
        r = self.client.post('/query/', data={'sql': 'SELECT 1'},
                             headers={'Sec-Fetch-Site': 'cross-site'})
        self.assertEqual(r.status_code, 403)
        r = self.client.post('/query/', data={'sql': 'SELECT 1'},
                             headers={'Sec-Fetch-Site': 'same-origin'})
        self.assertEqual(r.status_code, 200)
        r = self.client.post('/query/', data={'sql': 'SELECT 1'})
        self.assertEqual(r.status_code, 200)

    def test_get_does_not_execute_writes(self):
        self.client.get('/query/', query_string={'sql': 'DELETE FROM users'})
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM users')[0][0], 3)

    def test_post_runs_write_and_ddl(self):
        r = self.client.post('/query/',
                             data={'sql': "INSERT INTO users (username) "
                                          "VALUES ('x')"})
        self.assertIn(b'Rows modified', r.data)
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM users')[0][0], 4)
        self.client.post('/query/', data={'sql': 'CREATE TABLE t2 (id INT)'})
        self.assertEqual(self.client.get('/t2/').status_code, 200)

    def test_export_refuses_non_select(self):
        r = self.client.post('/query/', data={'sql': 'DROP TABLE users',
                                              'export_csv': '1'})
        self.assertIn(b'Only a single query may be exported', r.data)
        self.assertTrue(self.dbrows('SELECT COUNT(*) FROM users'))


class TestValueFilter(unittest.TestCase):
    def setUp(self):
        self._truncate = sw.app.config['TRUNCATE_VALUES']

    def tearDown(self):
        sw.app.config['TRUNCATE_VALUES'] = self._truncate

    def test_link_requires_full_match(self):
        url = 'https://example.com/x'
        self.assertEqual(sw.value_filter(url),
                         '<a href="%s">%s</a>' % (url, url))
        self.assertNotIn('<a ', sw.value_filter(url + ' trailing text'))

    def test_mailto(self):
        self.assertIn('<a href="mailto:huey@example.com"',
                      sw.value_filter('mailto:huey@example.com'))

    def test_long_link_label_truncated(self):
        url = 'https://example.com/' + 'x' * 60
        out = sw.value_filter(url)
        self.assertIn('href="%s"' % url, out)
        self.assertIn('...', out)

    def test_multiline_value_wrapped(self):
        self.assertEqual(sw.value_filter('line one\nline two'),
                         '<span class="pre">line one\nline two</span>')
        self.assertEqual(sw.value_filter('plain'), 'plain')

    def test_blob_respects_truncate_flag(self):
        data = b'\xff' * 600  # Undecodable, 1200 hex chars.
        sw.app.config['TRUNCATE_VALUES'] = True
        self.assertNotIn('ff' * 600, sw.value_filter(data))
        sw.app.config['TRUNCATE_VALUES'] = False
        self.assertIn('ff' * 600, sw.value_filter(data))


class TestFormatCreateTable(unittest.TestCase):
    def test_view_sql_unchanged(self):
        sql = ('CREATE VIEW v AS SELECT a FROM (SELECT a, b FROM t)\n'
               '  WHERE b = 1')
        self.assertEqual(sw.format_create_table(sql), sql)


class TestExplain(BaseAppTestCase):
    def test_explain_select(self):
        r = self.client.post('/query/', data={'sql': 'SELECT * FROM users',
                                              'explain': '1'})
        self.assertIn(b'SCAN', r.data)

    def test_explain_compiles_writes_without_running(self):
        self.client.post('/query/', data={
            'sql': "INSERT INTO users (username) VALUES ('x')",
            'explain': '1'})
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM users')[0][0], 3)

    def test_explain_suppresses_row_keys(self):
        # Plan rows have an "id" column, which must not become edit links.
        r = self.client.post('/users/query/', data={
            'sql': 'SELECT * FROM users', 'explain': '1'})
        self.assertNotIn(b'/users/update/', r.data)
        self.assertNotIn(b'/users/row/', r.data)
        self.assertNotIn(b'name="count"', r.data)


class TestPaginationGating(BaseAppTestCase):
    def test_read_paginates(self):
        r = self.client.post('/users/query/',
                             data={'sql': 'SELECT * FROM users'})
        self.assertIn(b'name="count"', r.data)
        self.assertIn(b'bulk-action', r.data)

    def test_returning_hides_pagination_and_bulk(self):
        # The count button and bulk form re-submit the sql. For a write
        # that would execute it again.
        r = self.client.post('/users/query/', data={
            'sql': "INSERT INTO users (username) VALUES ('r') RETURNING *"})
        self.assertNotIn(b'name="count"', r.data)
        self.assertNotIn(b'bulk-action', r.data)
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM users')[0][0], 4)

    def test_query_tab_bulk_delete(self):
        r = self.client.post('/users/query/', data={
            'sql': 'SELECT * FROM users', 'action': 'bulk-delete',
            'pk': key_encode([1])})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM users')[0][0], 2)
        self.assertIn(b'bulk-action', r.data)  # Fresh results offer bulk.


class TestForeignKeyLinks(BaseAppTestCase):
    def test_content_links_fk_values(self):
        r = self.client.get('/child/content/')
        self.assertIn(b'/parent/query/', r.data)

    def test_table_query_links_fk_values(self):
        r = self.client.post('/child/query/',
                             data={'sql': 'SELECT * FROM child'})
        self.assertIn(b'/parent/query/', r.data)

    def test_structure_shows_fk_target(self):
        r = self.client.get('/child/')
        self.assertIn(b'<code>parent.id</code>', r.data)

    def test_fk_link_resolves(self):
        r = self.client.get('/parent/query/', query_string={
            'sql': 'SELECT * FROM "parent" WHERE "id" = 1'})
        self.assertIn(b'p-one', r.data)

    def test_no_fk_links_on_generic_query(self):
        r = self.client.post('/query/', data={'sql': 'SELECT * FROM child'})
        self.assertNotIn(b'/parent/query/', r.data)


class TestLastViewed(BaseAppTestCase):
    def test_single_capped_session_key(self):
        with self.client.session_transaction() as s:
            s['users.last_viewed'] = [5, None]  # Legacy per-table key.
        self.client.get('/users/content/')
        self.client.get('/child/content/')
        with self.client.session_transaction() as s:
            self.assertNotIn('users.last_viewed', s)
            self.assertEqual([e[0] for e in s['last_viewed']],
                             ['child', 'users'])

    def test_back_links_carry_saved_position(self):
        # Real destination urls rendered into the hrefs, no bounce route.
        with self.client.session_transaction() as s:
            s['last_viewed'] = [['users', 3, -2]]
        for url in ('/users/row/%s/' % key_encode([1]),
                    '/users/update/%s/' % key_encode([1])):
            r = self.client.get(url)
            self.assertIn(b'page=3', r.data, url)
            self.assertIn(b'ordering=-2', r.data, url)
        # No saved entry falls back to plain content.
        r = self.client.get('/child/row/%s/' % key_encode([1]))
        self.assertIn(b'href="/child/content/"', r.data)

    def test_redirect_to_previous_uses_saved_position(self):
        with self.client.session_transaction() as s:
            s['last_viewed'] = [['users', 3, -2]]
        r = self.client.post('/users/delete/%s/' % key_encode([1]))
        self.assertIn(r.status_code, (302, 303))
        self.assertIn('page=3', r.headers['Location'])
        self.assertIn('ordering=-2', r.headers['Location'])


class TestRowKeyRoutes(BaseAppTestCase):
    def test_composite_key_with_delimiter(self):
        token = key_encode(['US', 'A:::B'])
        r = self.client.get('/comp/update/%s/' % token)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'composite-row', r.data)

    def test_blob_update_form_renders_hex(self):
        r = self.client.get('/blobs/update/%s/' % key_encode([b'\x00\xff']))
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'00ff', r.data)

    def test_blob_bulk_delete(self):
        token = key_encode([b'\x00\xff'])
        r = self.client.post('/blobs/content/',
                             data={'action': 'bulk-delete', 'pk': token},
                             headers={'Sec-Fetch-Site': 'same-origin'})
        self.assertIn(r.status_code, (302, 303))
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM blobs')[0][0], 0)

    def test_text_pk_valued_like_old_sentinel(self):
        self.client.post('/query/', data={'sql': "INSERT INTO users (id, "
                                          "username) VALUES (42, '__uneditable__')"})
        # A row whose value collides with the retired sentinel is still edited
        # by its own pk, not the username.
        token = key_encode([42])
        r = self.client.get('/users/update/%s/' % token)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'__uneditable__', r.data)

    def test_malformed_key_404s(self):
        self.assertEqual(self.client.get('/users/update/@@bad@@/').status_code,
                         404)

    def test_short_composite_token_cannot_multi_delete(self):
        # A one-value token against the two-column pk must be refused, a
        # zip would otherwise under-constrain the WHERE.
        token = key_encode(['US'])
        r = self.client.post('/comp/delete/%s/' % token)
        self.assertIn(r.status_code, (302, 303))
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM comp')[0][0], 1)
        r = self.client.get('/comp/row/%s/' % token)
        self.assertIn(r.status_code, (302, 303))

    def test_sanitized_pk_column_stays_in_key(self):
        # "user id" reflects as user_id and still keys the row. A grp-only
        # pk would target every row sharing grp.
        token = key_encode([7, 'a'])
        r = self.client.get('/oddpk/content/')
        self.assertIn(('/oddpk/row/%s/' % token).encode(), r.data)
        r = self.client.post('/oddpk/delete/%s/' % token)
        self.assertIn(r.status_code, (302, 303))
        self.assertEqual(self.dbrows('SELECT val FROM oddpk'), [('same-grp',)])


class TestDownload(BaseAppTestCase):
    def test_download_is_a_valid_snapshot(self):
        r = self.client.get('/download/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('attachment', r.headers['Content-Disposition'])
        self.assertIn('app.db', r.headers['Content-Disposition'])
        self.assertTrue(r.data.startswith(b'SQLite format 3\x00'))

        path = os.path.join(self.tmp, 'snapshot.db')
        with open(path, 'wb') as f:
            f.write(r.data)
        r.close()  # Fires call_on_close, which removes the temp snapshot.
        conn = sqlite3.connect(path)
        count, = conn.execute('SELECT COUNT(*) FROM users').fetchone()
        conn.close()
        self.assertEqual(count, 3)

    def test_temp_dir_removed_after_streaming(self):
        marker = os.path.join(self.tmp, 'dl-tmp')
        os.mkdir(marker)
        orig = sw.tempfile.mkdtemp
        sw.tempfile.mkdtemp = lambda: marker
        try:
            r = self.client.get('/download/')
            self.assertTrue(os.path.exists(marker))  # Held while streaming.
            self.assertTrue(r.data.startswith(b'SQLite format 3\x00'))
            self.assertFalse(os.path.exists(marker))  # Gone once consumed.
            r.close()
        finally:
            sw.tempfile.mkdtemp = orig

    def test_head_request_does_not_leak(self):
        # HEAD never starts the body generator, so cleanup must not depend
        # on the generator running.
        marker = os.path.join(self.tmp, 'dl-head')
        os.mkdir(marker)
        orig = sw.tempfile.mkdtemp
        sw.tempfile.mkdtemp = lambda: marker
        try:
            r = self.client.head('/download/')
            self.assertEqual(r.status_code, 200)
            self.assertIn('attachment', r.headers['Content-Disposition'])
            r.close()
            self.assertFalse(os.path.exists(marker))
        finally:
            sw.tempfile.mkdtemp = orig

    def test_temp_dir_removed_on_abandoned_download(self):
        # A client that disconnects mid-stream must not leak the snapshot.
        marker = os.path.join(self.tmp, 'dl-abandon')
        os.mkdir(marker)
        orig = sw.tempfile.mkdtemp
        sw.tempfile.mkdtemp = lambda: marker
        try:
            r = self.client.get('/download/')
            self.assertTrue(os.path.exists(marker))
            r.close()  # Closed without ever reading the body.
            self.assertFalse(os.path.exists(marker))
        finally:
            sw.tempfile.mkdtemp = orig

    def test_error_flashes_and_cleans_up(self):
        marker = os.path.join(self.tmp, 'dl-err')
        os.mkdir(marker)
        os.chmod(marker, 0o500)  # VACUUM INTO cannot create its file.
        orig = sw.tempfile.mkdtemp
        sw.tempfile.mkdtemp = lambda: marker
        try:
            r = self.client.get('/download/', follow_redirects=True)
            self.assertEqual(r.status_code, 200)
            self.assertIn(b'Error creating database snapshot', r.data)
            self.assertFalse(os.path.exists(marker))
        finally:
            sw.tempfile.mkdtemp = orig
            if os.path.exists(marker):
                os.chmod(marker, 0o700)


class TestMultiDb(BaseAppTestCase):
    def setUp(self):
        super().setUp()
        self.db2 = os.path.join(self.tmp, 'two.db')
        conn = sqlite3.connect(self.db2)
        conn.execute('CREATE TABLE t2 (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()
        sw.datasets.clear()
        sw.initialize_app([self.db_path, self.db2])
        self.client = sw.app.test_client()

    def test_download_follows_selected_dataset(self):
        self.client.get('/select-dataset/',
                        query_string={'name': os.path.realpath(self.db2)})
        r = self.client.get('/download/')
        self.assertIn('two.db', r.headers['Content-Disposition'])
        path = os.path.join(self.tmp, 'snap2.db')
        with open(path, 'wb') as f:
            f.write(r.data)
        r.close()
        conn = sqlite3.connect(path)
        tables = [t for t, in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        conn.close()
        self.assertEqual(tables, ['t2'])

        self.client.get('/select-dataset/',
                        query_string={'name': os.path.realpath(self.db_path)})
        r = self.client.get('/download/')
        self.assertIn('app.db', r.headers['Content-Disposition'])
        r.close()


class TestDuplicateBasenames(BaseAppTestCase):
    def setUp(self):
        super().setUp()
        self.db2 = os.path.join(self.tmp, 'alt', 'app.db')
        os.makedirs(os.path.dirname(self.db2))
        conn = sqlite3.connect(self.db2)
        conn.execute('CREATE TABLE t2 (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()
        sw.datasets.clear()
        sw.initialize_app([self.db_path, self.db2])
        sw.app.config['ENABLE_FILESYSTEM'] = True
        self.client = sw.app.test_client()

    def tearDown(self):
        super().tearDown()
        sw.app.config['ENABLE_FILESYSTEM'] = False

    def select(self, path):
        return self.client.get('/select-dataset/',
                               query_string={'name': os.path.realpath(path)})

    def test_both_databases_are_loaded(self):
        self.assertEqual(sorted(sw.datasets), sorted([
            os.path.realpath(self.db_path), os.path.realpath(self.db2)]))

    def test_each_database_is_selectable(self):
        self.select(self.db2)
        self.assertIn(b'href="/t2/"', self.client.get('/').data)

        self.select(self.db_path)
        self.assertIn(b'href="/users/"', self.client.get('/').data)

    def test_menu_shows_the_parent_directory(self):
        # Long paths stay in the link target, out of the menu text.
        r = self.client.get('/')
        self.assertIn(b'>alt/app.db<', r.data)
        self.assertIn(b'>%s/app.db<'
                      % os.path.basename(os.path.realpath(self.tmp)).encode(),
                      r.data)
        self.assertNotIn(b'>%s<' % os.path.realpath(self.db2).encode(), r.data)

    def test_download_uses_basename(self):
        self.select(self.db2)
        r = self.client.get('/download/')
        self.assertIn('app.db', r.headers['Content-Disposition'])
        r.close()

    def test_runtime_load_leaves_loaded_databases_alone(self):
        before = list(sw.datasets)
        third = os.path.join(self.tmp, 'other', 'app.db')
        os.makedirs(os.path.dirname(third))
        conn = sqlite3.connect(third)
        conn.execute('CREATE TABLE t3 (id INTEGER PRIMARY KEY)')
        conn.commit()
        conn.close()
        r = self.client.post('/load/', data={'mode': 'filesystem',
                                             'filename': third})
        self.assertIn(r.status_code, (302, 303))
        self.assertEqual(list(sw.datasets),
                         before + [os.path.realpath(third)])
        # The newly-loaded database becomes the selected one.
        self.assertIn(b'href="/t3/"', self.client.get('/').data)

    def test_unload_removes_only_the_named_database(self):
        r = self.client.post('/unload/',
                             data={'dataset': os.path.realpath(self.db2)})
        self.assertIn(r.status_code, (302, 303))
        self.assertEqual(list(sw.datasets), [os.path.realpath(self.db_path)])


class TestRowDetailEdges(BaseAppTestCase):
    def test_blob_pk_detail(self):
        r = self.client.get('/blobs/row/%s/' % key_encode([b'\x00\xff']))
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'blob', r.data)

    def test_extra_key_values_ignored(self):
        # Same behavior as update/delete, the first value drives the lookup.
        r = self.client.get('/users/row/%s/' % key_encode([1, 2]))
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'huey', r.data)

    def test_empty_key_404s(self):
        # A token holding no values, [] or a crafted {}, must 404 instead
        # of raising IndexError in decode_pk.
        for token in (key_encode([]), 'e30='):
            for route in ('row', 'update', 'delete'):
                url = '/users/%s/%s/' % (route, token)
                self.assertEqual(self.client.get(url).status_code, 404, url)

    def test_no_pk_table(self):
        r = self.client.get('/nopk/content/')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'no-pk-row', r.data)
        self.assertNotIn(b'/nopk/row/', r.data)
        r = self.client.get('/nopk/row/%s/' % key_encode(['no-pk-row']))
        self.assertIn(r.status_code, (302, 303))

    def test_sql_view(self):
        r = self.client.get('/v_users/content/')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'huey', r.data)
        self.assertNotIn(b'/v_users/row/', r.data)
        for route in ('row', 'update', 'delete'):
            r = self.client.get('/v_users/%s/%s/' % (route, key_encode([1])))
            self.assertIn(r.status_code, (302, 303), route)


class TestRowDetail(BaseAppTestCase):
    def test_detail_page(self):
        r = self.client.get('/users/row/%s/' % key_encode([1]))
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'huey', r.data)
        self.assertIn(b'/users/update/', r.data)

    def test_missing_row_redirects(self):
        r = self.client.get('/users/row/%s/' % key_encode([999]))
        self.assertIn(r.status_code, (302, 303))

    def test_malformed_key_404s(self):
        self.assertEqual(self.client.get('/users/row/@@bad@@/').status_code,
                         404)

    def test_composite_pk_detail(self):
        r = self.client.get('/comp/row/%s/' % key_encode(['US', 'A:::B']))
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'composite-row', r.data)

    def test_detail_links_fk_values(self):
        r = self.client.get('/child/row/%s/' % key_encode([1]))
        self.assertIn(b'/parent/query/', r.data)

    def test_view_links_on_content_and_query_tabs(self):
        self.assertIn(b'/users/row/', self.client.get('/users/content/').data)
        r = self.client.post('/users/query/',
                             data={'sql': 'SELECT * FROM users'})
        self.assertIn(b'/users/row/', r.data)

    def test_no_view_links_on_generic_query(self):
        r = self.client.post('/query/', data={'sql': 'SELECT * FROM users'})
        self.assertNotIn(b'/users/row/', r.data)


class TestReadOnlyRowDetail(BaseAppTestCase):
    def setUp(self):
        super().setUp()
        sw.datasets.clear()
        sw.initialize_app([self.db_path], read_only=True)
        self.client = sw.app.test_client()

    def tearDown(self):
        super().tearDown()
        sw.dataset_config['read_only'] = False

    def test_read_only_gets_view_but_not_edit(self):
        r = self.client.get('/users/content/')
        self.assertIn(b'/users/row/', r.data)
        self.assertNotIn(b'/users/update/', r.data)
        self.assertNotIn(b'toggle-pk-all', r.data)

    def test_read_only_detail_page(self):
        r = self.client.get('/users/row/%s/' % key_encode([2]))
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'mickey', r.data)
        self.assertNotIn(b'/users/update/', r.data)

    def test_read_only_download(self):
        # VACUUM INTO runs against the mode=ro connection.
        r = self.client.get('/download/')
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.data.startswith(b'SQLite format 3\x00'))
        r.close()

    def test_read_only_smoke(self):
        for url in ('/', '/users/', '/users/content/', '/users/query/'):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        r = self.client.get('/users/content/')
        self.assertNotIn(b'/users/insert/', r.data)
        r = self.client.post('/users/query/',
                             data={'sql': 'SELECT * FROM users'})
        self.assertIn(b'huey', r.data)

    def test_read_only_writes_fail_at_the_database(self):
        # Enforcement is the mode=ro connection. Writes must fail safely
        # with data unchanged and no 500s.
        r = self.client.post('/users/insert/', data={
            'chk_username': 'on', 'username': 'nope'},
            follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.dbrows('SELECT COUNT(*) FROM users')[0][0], 3)

        r = self.client.post('/users/drop/', follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertTrue(self.dbrows('SELECT COUNT(*) FROM users'))

        r = self.client.post('/users/update/%s/' % key_encode([1]),
                             data={'chk_username': 'on', 'username': 'x'},
                             follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        rows = self.dbrows('SELECT username FROM users WHERE id = 1')
        self.assertEqual(rows, [('huey',)])

        r = self.client.post('/create-table/', data={
            'table_name': 'zz', 'redirect': '/'}, follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'Error', r.data)


class TestPasswordlessLogin(BaseAppTestCase):
    def test_login_redirects_when_no_password(self):
        r = self.client.get('/login/')
        self.assertIn(r.status_code, (302, 303))
        r = self.client.post('/login/', data={})
        self.assertIn(r.status_code, (302, 303))
        with self.client.session_transaction() as s:
            self.assertNotIn('authorized', s)


class TestCreateTable(BaseAppTestCase):
    def test_create_failure_keeps_flash_destination(self):
        # The sqlite_ prefix is reserved, so creation fails. The redirect
        # must go back to the caller, not to a 404ing import page.
        r = self.client.post('/create-table/', data={
            'table_name': 'sqlite_nope', 'redirect': '/'},
            follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'Error', r.data)

    def test_create_success_lands_on_import(self):
        r = self.client.post('/create-table/', data={
            'table_name': 'fresh', 'redirect': '/'})
        self.assertIn('/fresh/import/', r.headers['Location'])


class TestErrorPages(BaseAppTestCase):
    def test_404_renders_in_chrome(self):
        r = self.client.get('/nope-not-a-table/')
        self.assertEqual(r.status_code, 404)
        self.assertIn(b'Not Found', r.data)
        self.assertIn(b'powered by', r.data)

    def test_403_renders_in_chrome(self):
        r = self.client.post('/query/', data={'sql': 'SELECT 1'},
                             headers={'Sec-Fetch-Site': 'cross-site'})
        self.assertEqual(r.status_code, 403)
        self.assertIn(b'powered by', r.data)

    def test_empty_registry_500_renders(self):
        # The error page renders even when no dataset can be resolved.
        sw.datasets.clear()
        r = self.client.get('/')
        self.assertEqual(r.status_code, 500)
        self.assertIn(b'powered by', r.data)


class TestQueryTemplates(BaseAppTestCase):
    def test_shared_form_renders_on_both_pages(self):
        for url, textarea_id in (('/query/', b'id="sql"'),
                                 ('/users/query/', b'id="table-sql"')):
            r = self.client.get(url)
            self.assertEqual(r.status_code, 200)
            self.assertIn(b'name="explain"', r.data)
            self.assertIn(b'id="bookmark-modal"', r.data)
            self.assertIn(b'id="import-bookmarks"', r.data)
            self.assertIn(b'__TABLE__', r.data)
            self.assertIn(b'id="sql-image-modal"', r.data)
            self.assertIn(textarea_id, r.data)

    def test_copy_affordances_present(self):
        r = self.client.get('/users/content/')
        self.assertIn(b'copy-row', r.data)
        self.assertIn(b'data-col="username"', r.data)

    def test_script_results_offer_copy(self):
        r = self.client.post('/users/query/',
                             data={'sql': 'SELECT 1; SELECT 2;'})
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'copy-row', r.data)
        self.assertNotIn(b'/users/row/', r.data)

    def test_missing_table_query_keeps_sql(self):
        # A bookmark can reference a dropped or foreign table. Its sql
        # falls back to the generic query page.
        r = self.client.get('/nope/query/', query_string={'sql': 'SELECT 1'})
        self.assertIn(r.status_code, (302, 303))
        self.assertIn('/query/?sql=SELECT', r.headers['Location'])
        r = self.client.get('/nope/query/', query_string={'sql': 'SELECT 1'},
                            follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'SELECT 1', r.data)

    def test_missing_table_query_404s_without_sql(self):
        self.assertEqual(self.client.get('/nope/query/').status_code, 404)


class TestInsertForm(BaseAppTestCase):
    def test_blank_numeric_inserts_null(self):
        # The form pre-enables every column, so a blank numeric input must
        # mean NULL instead of failing validation.
        r = self.client.post('/child/insert/', data={
            'chk_parent_id': 'on', 'parent_id': '',
            'chk_label': 'on', 'label': 'blank-num'})
        self.assertIn(r.status_code, (302, 303))
        rows = self.dbrows(
            'SELECT parent_id, label FROM child WHERE label = ?', 'blank-num')
        self.assertEqual(rows, [(None, 'blank-num')])

    def test_blank_text_inserts_empty_string(self):
        r = self.client.post('/child/insert/', data={
            'chk_label': 'on', 'label': ''})
        self.assertIn(r.status_code, (302, 303))
        rows = self.dbrows("SELECT COUNT(*) FROM child WHERE label = ''")
        self.assertEqual(rows, [(1,)])

    def test_blank_text_fk_keeps_empty_string(self):
        # A TEXT primary key can legitimately be '', so a blank input on
        # a text-keyed fk must not become NULL.
        r = self.client.post('/post/insert/', data={
            'chk_tag': 'on', 'tag': '',
            'chk_body': 'on', 'body': 'text-fk'})
        self.assertIn(r.status_code, (302, 303))
        rows = self.dbrows("SELECT tag FROM post WHERE body = 'text-fk'")
        self.assertEqual(rows, [('',)])


class TestContentTab(BaseAppTestCase):
    def test_renders_with_row_actions(self):
        r = self.client.get('/users/content/')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'/users/update/', r.data)
        self.assertIn(b'/users/delete/', r.data)
        self.assertIn(b'toggle-pk-all', r.data)

    def test_ordinal_ordering(self):
        asc = self.client.get('/users/content/',
                              query_string={'ordering': '2'}).data
        desc = self.client.get('/users/content/',
                               query_string={'ordering': '-2'}).data
        self.assertLess(asc.index(b'huey'), asc.index(b'zaizee'))
        self.assertLess(desc.index(b'zaizee'), desc.index(b'huey'))

    def test_bad_ordering_ignored(self):
        for value in ('99', 'abc', '-abc'):
            r = self.client.get('/users/content/',
                                query_string={'ordering': value})
            self.assertEqual(r.status_code, 200)

    def test_disabled_pager_arrows_are_not_links(self):
        # Single page of data, so all four arrows must render as spans.
        r = self.client.get('/users/content/')
        for arrow in (b'&laquo;', b'&lsaquo;', b'&rsaquo;', b'&raquo;'):
            self.assertIn(b'<span class="page-link">' + arrow, r.data)

    def test_flash_alerts_are_dismissible(self):
        r = self.client.get('/users/row/%s/' % key_encode([999]),
                            follow_redirects=True)
        self.assertIn(b'alert-dismissible', r.data)
        self.assertNotIn(b'alert-dismissable', r.data)


class TestUrlPrefix(BaseAppTestCase):
    def setUp(self):
        super(TestUrlPrefix, self).setUp()
        self.wsgi_app = sw.app.wsgi_app
        sw.initialize_app([], url_prefix='/sqlite/')

    def tearDown(self):
        sw.app.wsgi_app = self.wsgi_app
        sw.app.config['SESSION_COOKIE_PATH'] = None
        super(TestUrlPrefix, self).tearDown()

    def test_session_cookie_scoped_to_prefix(self):
        r = self.client.get('/sqlite/users/content/')
        self.assertIn('Path=/sqlite', r.headers['Set-Cookie'])


#
# Backfill: recipe parsing and normalization.
#

class TestParseRecipe(unittest.TestCase):
    def test_canonicalizes_steps(self):
        recipe = clean.parse_recipe(
            '[{"op": "case", "mode": "lower"}, '
            ' {"op": "replace", "pattern": "a+", "replacement": "A", '
            '  "flags": "IM"}, '
            ' {"op": "trim", "chars": " "}, '
            ' {"op": "null", "fill": "x"}]')
        self.assertEqual([step['op'] for step in recipe.steps],
                         ['case', 'replace', 'trim', 'null'])
        # Flags are de-duplicated and sorted, so equivalent recipes hash the
        # same and a preview cannot be invalidated by spelling.
        self.assertEqual(recipe.steps[1]['flags'], 'im')
        self.assertEqual(recipe.steps[3], {'op': 'null', 'fill': 'x'})
        again = clean.parse_recipe(
            '[{"op":"case","mode":"lower"},'
            '{"op":"replace","pattern":"a+","replacement":"A",'
            '"flags":"mi"},'
            '{"op":"trim","chars":" "},{"op":"null","fill":"x"}]')
        self.assertEqual(recipe.json, again.json)

    def test_rejects_bad_input(self):
        bad = [
            '',
            'not json',
            '{}',
            '["trim"]',
            '[1]',
            '[{"op": "wat"}]',
            '[{"op": "case"}]',
            '[{"op": "case", "mode": "sideways"}]',
            '[{"op": "replace", "pattern": "(", "replacement": ""}]',
            '[{"op": "replace", "pattern": "", "replacement": ""}]',
            '[{"op": "replace", "pattern": "a", "replacement": "", '
            '"flags": "q"}]',
            '[{"op": "trim", "extra": 1}]',
            '[{"op": "null", "fill": 2}]',
            '[{"op": "null", "fill": "x", "when": "always"}]',
            '[' + ','.join(['{"op": "trim"}'] * (clean.MAX_STEPS + 1)) + ']',
        ]
        for text in bad:
            with self.assertRaises(clean.CleanError, msg=text):
                clean.parse_recipe(text)


class TestApplyRecipe(unittest.TestCase):
    def recipe(self, *steps):
        import json
        return clean.parse_recipe(json.dumps(list(steps)))

    def test_ordered_pipeline(self):
        recipe = self.recipe(
            {'op': 'trim'},
            {'op': 'replace', 'pattern': r'(\d)\s+', 'replacement': r'\1-'},
            {'op': 'case', 'mode': 'upper'})
        self.assertEqual(clean.apply_recipe(recipe, '  a 1 b2 ')[0],
                         'A 1-B2')

    def test_null_is_not_the_text_null(self):
        recipe = self.recipe({'op': 'case', 'mode': 'upper'},
                             {'op': 'null', 'fill': 'X'})
        # NULL stays NULL through text steps, then the null step fills it.
        self.assertEqual(clean.apply_recipe(recipe, None), ('X', None))
        # The four characters N-U-L-L are just text.
        self.assertEqual(clean.apply_recipe(recipe, 'null'), ('NULL', None))
        no_fill = self.recipe({'op': 'case', 'mode': 'upper'})
        self.assertEqual(clean.apply_recipe(no_fill, None), (None, None))
        self.assertEqual(clean.apply_recipe(no_fill, 'null'), ('NULL', None))

    def test_null_when_modes(self):
        blank = self.recipe({'op': 'null', 'fill': 'B'})
        self.assertEqual(clean.apply_recipe(blank, '  ')[0], 'B')
        empty = self.recipe({'op': 'null', 'fill': 'E', 'when': 'empty'})
        self.assertEqual(clean.apply_recipe(empty, '  ')[0], '  ')
        self.assertEqual(clean.apply_recipe(empty, '')[0], 'E')
        null = self.recipe({'op': 'null', 'fill': 'N', 'when': 'null'})
        self.assertEqual(clean.apply_recipe(null, '')[0], '')
        self.assertEqual(clean.apply_recipe(null, None)[0], 'N')

    def test_blob_code_is_an_error(self):
        recipe = self.recipe({'op': 'trim'})
        value, error = clean.apply_recipe(recipe, b'\x00\xff')
        self.assertIsNone(value)
        self.assertIn('BLOB', error)

    def test_parse_mapping(self):
        self.assertEqual(
            clean.parse_mapping('category = grp\n# comment\n'
                                'weight -> qty'),
            [('category', 'grp'), ('weight', 'qty')])
        self.assertEqual(clean.parse_mapping('{"a": "b"}'), [('a', 'b')])
        self.assertEqual(clean.parse_mapping(''), [])
        for text in ('nonsense', 'x = a\ny = a', '= x', 'x ='):
            with self.assertRaises(clean.CleanError, msg=text):
                clean.parse_mapping(text)


#
# Backfill: planning and applying against a real database.
#

class BaseBackfillTestCase(BaseAppTestCase):
    SCHEMA = BaseAppTestCase.SCHEMA + """
        CREATE TABLE biz (id INTEGER PRIMARY KEY, code TEXT, grp TEXT,
            qty INTEGER, note TEXT);
        INSERT INTO biz (code, grp, qty, note) VALUES
            (' a1 ', NULL, NULL, 'one'),
            ('b2', 'old', 7, 'two'),
            ('zzz', NULL, NULL, 'three');
        CREATE TABLE dict (id INTEGER PRIMARY KEY, code TEXT, category TEXT,
            weight INTEGER);
        INSERT INTO dict (code, category, weight) VALUES
            ('A1', 'cat-a', 10), ('B2', 'cat-b', 20);
    """

    RECIPE = '[{"op": "trim"}, {"op": "case", "mode": "upper"}]'

    def config(self, **kwargs):
        values = {
            'table': 'biz',
            'code_column': 'code',
            'dict_table': 'dict',
            'dict_code_column': 'code',
            'mapping_text': 'category = grp\nweight = qty',
            'recipe_text': self.RECIPE,
            'update_code': True,
        }
        values.update(kwargs)
        return clean.parse_config(**values)

    def dataset(self):
        return sw.datasets[sw.datasets and list(sw.datasets)[0]]

    def biz_rows(self):
        return self.dbrows('SELECT * FROM biz ORDER BY id')


class TestBuildPlan(BaseBackfillTestCase):
    def test_preview_does_not_modify_business_rows(self):
        before = self.biz_rows()
        dataset = self.dataset()
        plan = clean.build_plan(dataset, self.config())
        self.assertEqual(self.biz_rows(), before)
        self.assertEqual(plan.counts['source_rows'], 3)
        self.assertEqual(plan.counts['matched_rows'], 2)
        self.assertEqual(plan.counts['unmatched_rows'], 1)
        self.assertEqual(plan.counts['error_rows'], 1)
        # The plan's rows inspect what SQLite would store, not the raw text.
        first = plan.rows[0]
        self.assertEqual(first.code_before, clean.Cell('text', ' a1 '))
        self.assertEqual(first.code_after, clean.Cell('text', 'A1'))

    def test_shows_values_sqlite_would_store(self):
        # "10" as text is stored as an integer in an INTEGER column, and the
        # preview must say so: applying it must not surprise anyone.
        plan = clean.build_plan(self.dataset(), self.config())
        row = [row for row in plan.rows if row.pk_label == '1'][0]
        self.assertEqual(row.after['qty'], clean.Cell('integer', 10))

    def test_no_match_and_ambiguous_match_are_errors(self):
        dataset = self.dataset()
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO dict (code, category, weight) "
                     "VALUES ('ZZZ', 'cat-z', 1)")
        conn.commit()
        conn.close()
        plan = clean.build_plan(dataset, self.config())
        # 'zzz' now matches exactly one row.
        self.assertEqual(plan.counts['unmatched_rows'], 0)

        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO dict (code, category, weight) "
                     "VALUES ('zzz', 'cat-z2', 2)")
        conn.commit()
        conn.close()
        plan = clean.build_plan(dataset, self.config())
        self.assertEqual(plan.counts['ambiguous_rows'], 1)
        row = [row for row in plan.error_rows
               if row.pk_label == '3'][0]
        self.assertIn('exactly one match is required', row.error)

        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM dict')
        conn.commit()
        conn.close()
        plan = clean.build_plan(dataset, self.config())
        self.assertEqual(plan.counts['unmatched_rows'], 3)

    def test_code_only_recipe(self):
        plan = clean.build_plan(
            self.dataset(), self.config(mapping_text=''))
        self.assertEqual(plan.updated_columns, ['code'])
        self.assertEqual(plan.counts['unmatched_rows'], 1)

    def test_requires_writable_target_and_pk(self):
        dataset = self.dataset()
        with self.assertRaises(clean.CleanError):
            clean.build_plan(dataset, self.config(code_column='missing'))
        with self.assertRaises(clean.CleanError):
            clean.build_plan(dataset, self.config(mapping_text='id = id'))
        with self.assertRaises(clean.CleanError):
            clean.build_plan(dataset, self.config(table='nopk'))
        with self.assertRaises(clean.CleanError):
            clean.build_plan(dataset, self.config(table='v_users'))
        with self.assertRaises(clean.CleanError):
            clean.build_plan(dataset, self.config(update_code=False,
                                                  mapping_text=''))

    def test_duplicate_dictionary_codes_are_counted(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO dict (code, category, weight) "
                     "VALUES (' a1 ', 'cat-a2', 11)")
        conn.commit()
        conn.close()
        plan = clean.build_plan(self.dataset(), self.config())
        self.assertEqual(plan.counts['dictionary_duplicate_codes'], 1)
        self.assertEqual(plan.counts['ambiguous_rows'], 1)


class TestApplyPlan(BaseBackfillTestCase):
    def test_apply_updates_only_selected_columns_by_pk(self):
        dataset = self.dataset()
        before = self.biz_rows()
        config = self.config()
        # Drop the row that cannot match so the batch is clean.
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM biz WHERE code = 'zzz'")
        conn.commit()
        conn.close()
        before = self.biz_rows()

        plan = clean.build_plan(dataset, config)
        result = clean.apply_plan(dataset, config, plan.fingerprint)
        self.assertEqual(result.applied, 2)
        after = self.biz_rows()
        self.assertEqual([(r[0], r[1], r[2], r[3], r[4]) for r in after],
                         [(1, 'A1', 'cat-a', 10, 'one'),
                          (2, 'B2', 'cat-b', 20, 'two')])
        # Primary keys and unselected columns are untouched; only the
        # mapped columns and the code differ from before.
        for old, new in zip(before, after):
            self.assertEqual(old[0], new[0])
            self.assertEqual(old[4], new[4])

    def test_stale_preview_is_refused(self):
        dataset = self.dataset()
        config = self.config()
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM biz WHERE code = 'zzz'")
        conn.commit()
        conn.close()
        plan = clean.build_plan(dataset, config)

        # A dictionary edit invalidates the preview.
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE dict SET category = 'changed' "
                     "WHERE code = 'A1'")
        conn.commit()
        conn.close()
        before = self.biz_rows()
        with self.assertRaises(clean.StalePreview):
            clean.apply_plan(dataset, config, plan.fingerprint)
        self.assertEqual(self.biz_rows(), before)

        # A recipe edit invalidates it as well.
        fresh = clean.build_plan(dataset, config)
        edited = self.config(recipe_text='[{"op": "trim"}]')
        with self.assertRaises(clean.StalePreview):
            clean.apply_plan(dataset, edited, fresh.fingerprint)
        self.assertEqual(self.biz_rows(), before)

        # A missing fingerprint means "not previewed".
        with self.assertRaises(clean.StalePreview):
            clean.apply_plan(dataset, config, '')
        self.assertEqual(self.biz_rows(), before)

    def test_source_and_schema_changes_invalidate_the_preview(self):
        dataset = self.dataset()
        config = self.config()
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM biz WHERE code = 'zzz'")
        conn.commit()
        conn.close()
        plan = clean.build_plan(dataset, config)

        # A business-row edit (here: a changed unselected column) is enough.
        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE biz SET note = 'edited' WHERE id = 1")
        conn.commit()
        conn.close()
        before = self.biz_rows()
        with self.assertRaises(clean.StalePreview):
            clean.apply_plan(dataset, config, plan.fingerprint)
        self.assertEqual(self.biz_rows(), before)

        # A new row is a source change too.
        fresh = clean.build_plan(dataset, config)
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO biz (code, note) VALUES ('a1', 'new')")
        conn.commit()
        conn.close()
        with self.assertRaises(clean.StalePreview):
            clean.apply_plan(dataset, config, fresh.fingerprint)

        # A table-structure change invalidates it; the re-plan then fails
        # loudly because the column is gone.
        fresh = clean.build_plan(dataset, config)
        conn = sqlite3.connect(self.db_path)
        conn.execute('ALTER TABLE biz ADD COLUMN extra TEXT')
        conn.commit()
        conn.close()
        with self.assertRaises(clean.StalePreview):
            clean.apply_plan(dataset, config, fresh.fingerprint)
        self.assertEqual(
            self.dbrows('SELECT note FROM biz ORDER BY id')[0], ('edited',))

    def test_errors_refuse_the_whole_batch(self):
        dataset = self.dataset()
        before = self.biz_rows()
        plan = clean.build_plan(dataset, self.config())
        self.assertTrue(plan.error_count)
        with self.assertRaises(clean.PlanRefused):
            clean.apply_plan(dataset, self.config(), plan.fingerprint)
        self.assertEqual(self.biz_rows(), before)

    def test_constraint_error_rolls_back_everything(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            DROP TABLE biz;
            CREATE TABLE biz (id INTEGER PRIMARY KEY, code TEXT, grp TEXT,
                qty INTEGER CHECK (qty < 15), note TEXT);
            INSERT INTO biz (code, grp, qty, note) VALUES
                (' a1 ', NULL, NULL, 'one'),
                ('b2', NULL, NULL, 'two');
        """)
        conn.commit()
        conn.close()
        dataset = self.dataset()
        config = self.config()
        plan = clean.build_plan(dataset, config)
        # The CHECK constraint is not modeled by the probe, so planning
        # succeeds and the failure surfaces while applying.  The first row
        # (qty 10) would pass; the second (qty 20) fails, proving the whole
        # batch rolls back including the already-applied first update.
        self.assertEqual(plan.counts['error_rows'], 0)
        before = self.biz_rows()
        with self.assertRaises(clean.CleanError) as ctx:
            clean.apply_plan(dataset, config, plan.fingerprint)
        self.assertIn('rolled back', str(ctx.exception))
        self.assertEqual(self.biz_rows(), before)

    def test_strict_table_conversion_error_is_previewed(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            DROP TABLE biz;
            CREATE TABLE biz (id INTEGER PRIMARY KEY, code TEXT, grp TEXT,
                qty INTEGER, note TEXT) STRICT;
            INSERT INTO biz (code, grp, qty, note) VALUES
                (' a1 ', NULL, NULL, 'one'),
                ('b2', NULL, NULL, 'two');
            UPDATE dict SET weight = 'not-a-number' WHERE code = 'B2';
        """)
        conn.commit()
        conn.close()
        dataset = self.dataset()
        config = self.config()
        plan = clean.build_plan(dataset, config)
        self.assertEqual(plan.counts['error_rows'], 1)
        row = plan.error_rows[0]
        self.assertEqual(row.pk_label, '2')
        self.assertIn('qty', row.error)
        before = self.biz_rows()
        with self.assertRaises(clean.PlanRefused):
            clean.apply_plan(dataset, config, plan.fingerprint)
        self.assertEqual(self.biz_rows(), before)

    def test_without_rowid_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            DROP TABLE biz;
            CREATE TABLE biz (code TEXT, id TEXT, grp TEXT, qty INTEGER,
                note TEXT, PRIMARY KEY (id)) WITHOUT ROWID;
            INSERT INTO biz VALUES (' a1 ', 'x', NULL, NULL, 'one'),
                                   ('b2', 'y', NULL, NULL, 'two');
        """)
        conn.commit()
        conn.close()
        dataset = self.dataset()
        config = self.config()
        plan = clean.build_plan(dataset, config)
        self.assertEqual(plan.error_count, 0)
        self.assertEqual(clean.apply_plan(dataset, config,
                                          plan.fingerprint).applied, 2)
        self.assertEqual(
            self.dbrows('SELECT code, id, grp, qty, note FROM biz '
                        'ORDER BY id'),
            [('A1', 'x', 'cat-a', 10, 'one'),
             ('B2', 'y', 'cat-b', 20, 'two')])

    def test_primary_key_columns_are_immutable(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            DROP TABLE biz;
            CREATE TABLE biz (code TEXT PRIMARY KEY, grp TEXT,
                qty INTEGER, note TEXT) WITHOUT ROWID;
            INSERT INTO biz VALUES (' a1 ', NULL, NULL, 'one');
        """)
        conn.commit()
        conn.close()
        dataset = self.dataset()
        # Rewriting a code that is part of the primary key would change the
        # key, which the feature promises never to do.
        with self.assertRaises(clean.CleanError) as ctx:
            clean.build_plan(dataset, self.config())
        self.assertIn('primary key', str(ctx.exception))
        # Mapping a different column still works with the code untouched.
        plan = clean.build_plan(
            dataset, self.config(update_code=False))
        self.assertEqual(plan.counts['error_rows'], 0)

    def test_null_text_and_blob_stay_distinct(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            DROP TABLE biz;
            CREATE TABLE biz (id INTEGER PRIMARY KEY, code, grp TEXT,
                note TEXT);
            INSERT INTO biz (code, note) VALUES
                (x'4142', 'blob'), ('NULL', 'text-null'), (NULL, 'sqlnull'),
                (' a1 ', 'match');
            INSERT INTO dict (code, category, weight) VALUES
                ('NULL', 'cat-textnull', 1);
        """)
        conn.commit()
        conn.close()
        dataset = self.dataset()
        config = self.config(mapping_text='category = grp')
        plan = clean.build_plan(dataset, config)
        by_pk = {row.pk_label: row for row in plan.rows}
        self.assertEqual(by_pk['1'].code_before, clean.Cell('blob', b'AB'))
        self.assertEqual(by_pk['2'].code_before, clean.Cell('text', 'NULL'))
        self.assertEqual(by_pk['3'].code_before, clean.Cell('null'))
        # The BLOB and the NULL had no dictionary match/recipe.
        self.assertIn('BLOB', by_pk['1'].error)
        self.assertIn('NULL', by_pk['3'].error)
        # The text "NULL" matched a dictionary row with code 'NULL'.
        self.assertEqual(by_pk['2'].match_state, 'ok')
        self.assertEqual(by_pk['4'].code_after, clean.Cell('text', 'A1'))

        # Applying keeps the BLOB as a BLOB and the text NULL as text.
        conn = sqlite3.connect(self.db_path)
        conn.execute('DELETE FROM biz WHERE id IN (1, 3)')
        conn.commit()
        conn.close()
        plan = clean.build_plan(dataset, config)
        self.assertEqual(plan.error_count, 0)
        clean.apply_plan(dataset, config, plan.fingerprint)
        rows = self.dbrows('SELECT id, typeof(code), code FROM biz '
                           'ORDER BY id')
        self.assertEqual(rows[0][1:], ('text', 'NULL'))
        self.assertEqual(rows[1][1:], ('text', 'A1'))


class TestCleanRoute(BaseBackfillTestCase):
    def preview(self, **overrides):
        data = {
            'code_column': 'code',
            'dict_table': 'dict',
            'dict_code_column': 'code',
            'mapping': 'category = grp\nweight = qty',
            'recipe': self.RECIPE,
            'update_code': '1',
            'action': 'preview',
        }
        data.update(overrides)
        return self.client.post('/biz/clean/', data=data)

    def fingerprint(self, html):
        import re as _re
        match = _re.search(rb'name="fingerprint" type="hidden" '
                           rb'value="([0-9a-f]+)"', html)
        self.assertIsNotNone(match)
        return match.group(1).decode('ascii')

    def test_clean_tab_and_form(self):
        r = self.client.get('/biz/clean/')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'Business code column', r.data)
        self.assertIn(b'Clean', self.client.get('/biz/').data)

    def test_preview_leaves_the_database_alone(self):
        before = self.biz_rows()
        r = self.preview()
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.biz_rows(), before)
        self.assertIn(b'No dictionary row matches', r.data)
        # The preview distinguishes SQL NULL from the text NULL and BLOB.
        self.assertIn(b'cell-null', r.data)

    def test_apply_roundtrip_and_stale_refusal(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM biz WHERE code = 'zzz'")
        conn.commit()
        conn.close()
        before = self.biz_rows()
        r = self.preview()
        fingerprint = self.fingerprint(r.data)
        data = {
            'code_column': 'code',
            'dict_table': 'dict',
            'dict_code_column': 'code',
            'mapping': 'category = grp\nweight = qty',
            'recipe': self.RECIPE,
            'update_code': '1',
            'action': 'apply',
            'fingerprint': fingerprint,
        }
        # Applying with a bogus fingerprint refuses and leaves rows intact.
        stale = dict(data, fingerprint='nope')
        r = self.client.post('/biz/clean/', data=stale)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'changed since the preview', r.data)
        self.assertEqual(self.biz_rows(), before)

        r = self.client.post('/biz/clean/', data=data, follow_redirects=True)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'Successfully updated 2 row(s)', r.data)
        self.assertEqual(
            self.dbrows('SELECT id, code, grp, qty, note FROM biz '
                        'ORDER BY id'),
            [(1, 'A1', 'cat-a', 10, 'one'),
             (2, 'B2', 'cat-b', 20, 'two')])

    def test_apply_refused_when_rows_have_errors(self):
        before = self.biz_rows()
        r = self.preview()
        self.assertIn(b'Apply changes', r.data)
        # The button is disabled client-side, but a crafted POST must still
        # be refused server-side.
        data = {
            'code_column': 'code',
            'dict_table': 'dict',
            'dict_code_column': 'code',
            'mapping': 'category = grp\nweight = qty',
            'recipe': self.RECIPE,
            'update_code': '1',
            'action': 'apply',
            'fingerprint': self.fingerprint(r.data),
        }
        r = self.client.post('/biz/clean/', data=data)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'nothing was updated', r.data)
        self.assertEqual(self.biz_rows(), before)

    def test_view_is_rejected(self):
        data = {
            'code_column': 'id',
            'dict_table': 'dict',
            'dict_code_column': 'code',
            'mapping': '',
            'recipe': self.RECIPE,
            'action': 'preview',
        }
        r = self.client.post('/v_users/clean/', data=data)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'is a view', r.data)
        self.assertNotIn(b'/v_users/clean/', self.client.get(
            '/v_users/structure/').data)

    def test_unknown_table_404s(self):
        self.assertEqual(self.client.get('/nope/clean/').status_code, 404)

    def test_bad_recipe_shows_error(self):
        r = self.preview(recipe='[{"op": "nope"}]')
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'unknown op', r.data)


class TestCleanReadOnly(BaseBackfillTestCase):
    def setUp(self):
        super(TestCleanReadOnly, self).setUp()
        sw.datasets.clear()
        sw.initialize_app([self.db_path], read_only=True)
        self.client = sw.app.test_client()

    def tearDown(self):
        super(TestCleanReadOnly, self).tearDown()
        sw.dataset_config['read_only'] = False

    def test_preview_works_apply_is_disabled(self):
        data = {
            'code_column': 'code',
            'dict_table': 'dict',
            'dict_code_column': 'code',
            'mapping': 'category = grp',
            'recipe': self.RECIPE,
            'update_code': '1',
            'action': 'preview',
        }
        before = self.biz_rows()
        r = self.client.post('/biz/clean/', data=data)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'read-only', r.data)
        self.assertNotIn(b'name="action" type="submit" value="apply"',
                         r.data)
        self.assertEqual(self.biz_rows(), before)

        # A crafted apply is refused at the application layer.
        data.update(action='apply', fingerprint='x')
        r = self.client.post('/biz/clean/', data=data)
        self.assertEqual(r.status_code, 200)
        self.assertIn(b'read-only', r.data)
        self.assertEqual(self.biz_rows(), before)


if __name__ == '__main__':
    unittest.main()
