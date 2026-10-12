import csv
from pathlib import Path
import shutil
import tempfile
import unittest
from PIL import Image, PngImagePlugin
from test_data_split import candidates
from extensions_built_in.streetscape_cnn.data import prepare_dataset
from extensions_built_in.streetscape_cnn.contract import file_sha256, read_json


class DataPreparationTests(unittest.TestCase):
    def fixture(self, root):
        source, frozen = root / 'source', root / 'frozen'
        source.mkdir()
        frozen.mkdir()
        rows = candidates()
        for index, row in enumerate(rows):
            name = row['image_id']
            original = frozen / f'{name}.png'
            picture = Image.new('RGB', (4, 4), (index % 256, index // 256, index % 127))
            picture.save(original)
            shutil.copy2(original, source / original.name)
            (source / f'{name}.txt').write_text('A street with buildings and pedestrians.', encoding='utf-8')
            row.update(source_path=str(original), sha256=file_sha256(original))
        csv_path = root / 'candidates.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return source, csv_path, rows

    def test_lossless_export_and_repeat_preserve_all_source_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, csv_path, rows = self.fixture(root)
            image = source / f'{rows[0]["image_id"]}.png'
            with Image.open(image) as picture:
                info = PngImagePlugin.PngInfo()
                info.add_text('export', 'tagger')
                picture.save(root / 'export.png', pnginfo=info)
            shutil.copy2(root / 'export.png', image)
            before = {p.name: file_sha256(p) for p in source.iterdir()}
            output = root / 'output'
            result = prepare_dataset(source, csv_path, output)
            self.assertEqual(result, prepare_dataset(source, csv_path, output))
            manifest = read_json(output / 'dataset_manifest.json')
            self.assertEqual(manifest['pairs'][0]['pixel_provenance'], 'tagger_lossless_reencoding_pixel_verified')
            self.assertEqual(before, {p.name: file_sha256(p) for p in source.iterdir()})

    def test_pixel_changes_missing_files_and_target_conflicts_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, csv_path, rows = self.fixture(root)
            first = source / f'{rows[0]["image_id"]}.png'
            Image.new('RGB', (4, 4), (100, 100, 100)).save(first)
            with self.assertRaises(ValueError):
                prepare_dataset(source, csv_path, root / 'output')
            shutil.copy2(rows[0]['source_path'], first)
            caption = first.with_suffix('.txt')
            text = caption.read_text()
            caption.unlink()
            with self.assertRaises(ValueError):
                prepare_dataset(source, csv_path, root / 'output')
            caption.write_text(text, encoding='utf-8')
            prepare_dataset(source, csv_path, root / 'output')
            row = read_json(root / 'output/dataset_manifest.json')['pairs'][0]
            target = root / 'output' / row['image']
            target.write_bytes(b'conflicting content')
            with self.assertRaises(ValueError):
                prepare_dataset(source, csv_path, root / 'output')
            self.assertEqual(target.read_bytes(), b'conflicting content')
