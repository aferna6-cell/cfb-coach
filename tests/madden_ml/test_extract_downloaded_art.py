"""Offline synthetic fixtures only; downloader-owned files must be byte-identical."""
import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('downloaded_art', ROOT / 'scripts/madden_ml/extract_downloaded_art.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def image(raw, name='offense__fixture-formation__fixture-play.png'):
    raw.mkdir(exist_ok=True)
    path = raw / name
    Image.new('RGB', (556, 258), 'black').save(path)
    return path


def test_resumable_hash_and_no_catalog_or_authorization_writes(tmp_path, monkeypatch):
    catalog = tmp_path / 'catalog.jsonl'
    auth = tmp_path / 'authorization.json'
    catalog.write_text('catalog sentinel')
    auth.write_text('authorization sentinel')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (catalog, auth)}
    path = image(tmp_path / 'raw')
    out = tmp_path / 'extractions'
    assert runner.process(path, out, set(), True) == 'extracted'
    dest, summary, _ = runner.paths(out, runner.identity(path))
    assert dest.exists() and summary.exists()
    assert not list(out.glob('*overlay.png'))
    assert runner.load(summary)['tier'] == 'diagram_extracted_unreviewed'
    assert runner.load(summary)['model_feature_eligible'] is False
    analyze = runner.pe.analyze
    monkeypatch.setattr(runner.pe, 'analyze', lambda *a: (_ for _ in ()).throw(AssertionError('must skip')))
    assert runner.process(path, out, set()) == 'skipped'
    summary.unlink()
    assert runner.process(path, out, set()) == 'skipped'
    assert summary.exists()
    monkeypatch.setattr(runner.pe, 'analyze', analyze)
    Image.new('RGB', (556, 258), 'gray').save(path)
    assert runner.process(path, out, set()) == 'extracted'
    for p, expected in before.items():
        assert (p.read_bytes(), p.stat().st_mtime_ns) == expected


def test_exact_filename_and_processing_order(tmp_path):
    names = ['offense__other__one.png', 'defense__other__one.png',
             'defense__installed__one.png', 'offense__installed__one.png']
    installed = {('offense', 'installed', 'one'), ('defense', 'installed', 'one')}
    for name in names:
        image(tmp_path, name)
    assert [p.name for p in runner.ordered_files(tmp_path, installed)] == list(reversed(names))
    for bad in ['offense__f.png', 'mystery__f__p.png', 'offense__f__p__q.png', 'offense__F__p.png']:
        path = tmp_path / bad
        assert runner.identity(path) is None
        assert runner.process(path, tmp_path / 'out', installed) == 'unmatched_filename'


def test_failed_partial_png_retried_after_bytes_change(tmp_path):
    path = tmp_path / 'defense__fixture__play.png'
    path.write_bytes(b'partial')
    out = tmp_path / 'extractions'
    assert runner.process(path, out, set()) == 'failed'
    Image.new('RGB', (556, 258), 'black').save(path)
    assert runner.process(path, out, set()) == 'extracted'


def test_no_defender_marker_matching_or_colour_meaning_invented():
    ex = {'extractor': runner.pe.EXTRACTOR_VERSION, 'flags': [],
          'players': [{'index_left_to_right': 0, 'shape': 'square', 'x': 100, 'y': 200}],
          'zones': [{'colour': 'blue', 'bbox': [0, 0, 200, 100]}],
          'rush_arrows': [{'start': [100, 200], 'end': [100, 220]}]}
    summary = runner.proposed_responsibilities(ex, ('defense', 'fixture', 'one'), 'hash')
    assert summary['markers'][0]['slot'] == 'unknown'
    assert summary['markers'][0]['matched'] == 'unknown'
    assert summary['zones_unassociated'] == ex['zones']
    assert 'UNVERIFIED' in summary['colour_meaning']
    assert summary['rush_arrows_unassociated'][0]['interpretation'] == 'unverified_extractor_proposal'


def test_installed_inventory_only_preapproved_aliases():
    rows = runner.installed_inventory()
    assert len(rows) == 216
    assert next(r for r in rows if r['play'] == 'L Spot')['identity'] == ('offense', 'gun-5wr-tight', 'z-spot')
    assert sum(r['identity_basis'] == 'owner_approved_site_alias' for r in rows) == 2


def test_output_identity_boundaries_cannot_collide(tmp_path):
    assert runner.paths(tmp_path, ('offense', 'a-b', 'c'))[0] != runner.paths(tmp_path, ('offense', 'a', 'b-c'))[0]


def test_report_counts_missing_images_and_refuses_store_writes(tmp_path):
    import pytest
    store = tmp_path / 'store'
    store.mkdir(exist_ok=True)
    path = image(store / 'raw')
    ident = runner.identity(path)
    inv = [{'side': 'offense', 'formation': 'Fixture Formation', 'play': 'Fixture Play', 'identity': ident},
           {'side': 'defense', 'formation': 'Missing', 'play': 'Missing', 'identity': ('defense', 'missing', 'missing')}]
    assert runner.process(path, store / 'extractions', {ident}) == 'extracted'
    counts = runner.make_report(store, inv, tmp_path / 'progress')
    assert counts['installed']['offense']['extracted'] == 1
    assert counts['installed']['defense']['downloaded_with_art'] == 0
    assert 'image missing' not in (tmp_path / 'report.md').read_text()  # only offense table contains missing-image labels
    assert 'Missing / Missing' in (tmp_path / 'report.md').read_text()
    with pytest.raises(ValueError, match='outside'):
        runner.make_report(store, inv, store / 'authorization')
