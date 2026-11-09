"""Dataset viewer contract: pairing, broken data, filters and HTTP access."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

from apps.sd1_viewer.server import Dataset, make_server


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('imgs/A', 'imgs/B', 'Annotations/A', 'Annotations/B'):
            (self.root / folder).mkdir(parents=True)
        for name in ('A/same.JPG', 'B/same.jpg', 'B/plain.png'):
            (self.root / 'imgs' / name).write_bytes(b'image')
        xml = '<annotation><size><width>100</width><height>80</height></size><object><name>scratch</name><bndbox><xmin>10</xmin><ymin>5</ymin><xmax>40</xmax><ymax>30</ymax></bndbox></object></annotation>'
        (self.root / 'Annotations/A/same.xml').write_text(xml)
        (self.root / 'Annotations/B/same.xml').write_text('<broken')
        (self.root / 'Annotations/A/orphan.xml').write_text(xml)
        self.dataset = Dataset(self.root)

    def test_pairing_retains_orphans_and_invalid_annotations(self):
        stats = self.dataset.summary()
        self.assertEqual(stats['images'], 3)
        self.assertEqual(stats['paired'], 2)
        self.assertEqual(stats['annotation_only'], 1)
        self.assertEqual(stats['invalid'], 1)
        self.assertEqual(stats['labels']['scratch'], 2)

    def test_filters_include_missing_images_and_class(self):
        rows = self.dataset.query(label='scratch')['items']
        self.assertEqual(len(rows), 2)
        orphan = self.dataset.query(status='annotation_only')['items'][0]
        self.assertEqual(orphan['key'], 'A/orphan')
        self.assertIsNone(self.dataset.detail(orphan['id'])['image_url'])
        self.assertEqual(self.dataset.query(q='B/same')['total'], 1)

    def test_http_serves_only_indexed_images(self):
        server = make_server(self.dataset, '127.0.0.1', 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f'http://127.0.0.1:{server.server_port}'
        with urlopen(base + '/api/items?status=paired') as response:
            data = json.load(response)
        self.assertEqual(data['total'], 2)
        item = self.dataset.detail(data['items'][0]['id'])
        with urlopen(base + item['image_url']) as response:
            self.assertEqual(response.read(), b'image')
        with self.assertRaises(HTTPError) as caught:
            urlopen(base + '/image/../../secret')
        self.assertEqual(caught.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
