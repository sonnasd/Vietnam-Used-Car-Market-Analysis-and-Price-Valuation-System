
import argparse
import csv
import json
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ENDPOINT = 'https://gateway.chotot.com/v1/public/ad-listing'
FIELDS = ('list_id ad_id subject price price_string category category_name condition_ad '
          'condition_ad_name carbrand carbrand_name carmodel carmodel_name mfdate '
          'mileage_v2 gearbox fuel carseats cartype carcolor carorigin region_v2 '
          'region_name region_name_v3 area_name ward_name list_time orig_list_time '
          'image images number_of_images type status').split()

def fetch(params):
    url = ENDPOINT + '?' + urlencode(params)
    for attempt in range(5):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'UsedCarResearch/1.0', 'Accept': 'application/json'}), timeout=45) as response:
                result = json.load(response)
            if not isinstance(result.get('ads'), list):
                raise ValueError('Response does not contain ads array')
            return result, url
        except HTTPError as exc:
            if exc.code == 429:
                delay = exc.headers.get('Retry-After', '60')
                delay = int(delay) if delay.isdigit() else 60
                print(f'Rate limit; waiting {delay}s', flush=True)
                time.sleep(max(60, delay))
            elif exc.code >= 500 and attempt < 4:
                time.sleep(5 * (attempt + 1))
            else:
                raise
        except (URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError('Retry limit reached')

def export(db, out, target):
    rows = [json.loads(r[0]) for r in db.execute('SELECT payload FROM cars ORDER BY rowid')]
    with (out / 'cars.jsonl').open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    # CSV is a flat interchange export, not a formatted workbook.
    with (out / 'cars.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ['source_url', 'fetched_at'])
        writer.writeheader()
        for row in rows:
            row = dict(row)
            for key, value in row.items():
                if isinstance(value, list):
                    row[key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')):
                    row[key] = "'" + value
            writer.writerow(row)
    report = {
        'target': target, 'count': len(rows), 'unique_list_ids': len({r['list_id'] for r in rows}),
        'all_used_cars': all(r['category'] == 2010 and r['condition_ad'] == 1 for r in rows),
        'regions': dict(Counter(r.get('region_name_v3') or r.get('region_name') for r in rows)),
        'missing': {key: sum(r.get(key) is None for r in rows) for key in ['price','mfdate','mileage_v2','carbrand_name','carmodel_name']},
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'note': 'Nationwide price-partitioned collection; unique listing IDs, not necessarily unique physical vehicles. Categorical codes retained as returned by API.'
    }
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=15000)
    parser.add_argument('--delay', type=float, default=1.5)
    parser.add_argument('--out', default='data')
    args = parser.parse_args()
    if args.target < 1 or args.delay < 1:
        parser.error('target >= 1 and delay >= 1 required')
    out = Path(__file__).resolve().parent / args.out
    out.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / 'cars.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS cars (list_id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS progress (source TEXT PRIMARY KEY, offset INTEGER, done INTEGER)')
    total = db.execute('SELECT COUNT(*) FROM cars').fetchone()[0]
    # Small price bands avoid the 10,000-result query ceiling. All are nationwide.
    bounds = [0, 100000000, 200000000, 300000000, 400000000, 500000000,
              600000000, 800000000, 1000000000, 1500000000, 2500000000, 1000000000000]
    try:
        for low, high in zip(bounds, bounds[1:]):
            if total >= args.target:
                break
            source = f'{low}-{high - 1}'
            checkpoint = db.execute('SELECT offset,done FROM progress WHERE source=?', (source,)).fetchone()
            offset, done = checkpoint if checkpoint else (0, 0)
            if done:
                continue
            stalled = 0
            while total < args.target:
                time.sleep(args.delay)
                result, url = fetch({'cg':2010, 'condition_ad':1, 'st':'s,k', 'limit':50, 'o':offset, 'price':source})
                ads = result['ads']
                if offset == 0:
                    print(f'PRICE {source} API total={result.get("total")}', flush=True)
                if not ads:
                    with db:
                        db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,1)', (source,offset))
                    break
                if any(a.get('category') != 2010 or a.get('condition_ad') != 1 or not a.get('list_id') or not low <= a.get('price', -1) < high for a in ads):
                    raise ValueError('API returned an invalid category, condition, ID or price; stopping')
                new = 0
                consumed = 0
                now = datetime.now(timezone.utc).isoformat()
                with db:
                    for ad in ads:
                        if total >= args.target:
                            break
                        row = {key: ad.get(key) for key in FIELDS}
                        row.update(source_url=url, fetched_at=now)
                        added = db.execute('INSERT OR IGNORE INTO cars VALUES (?,?)', (ad['list_id'], json.dumps(row, ensure_ascii=False))).rowcount
                        total += added
                        new += added
                        consumed += 1
                    offset += consumed
                    db.execute('INSERT OR REPLACE INTO progress VALUES (?,?,0)', (source,offset))
                print(f'price={source} offset={offset} new={new} total={total}/{args.target}', flush=True)
                stalled = stalled + 1 if new == 0 else 0
                if stalled >= 5:
                    raise RuntimeError('Five pages without new IDs; inspect pagination before continuing')
    finally:
        export(db, out, args.target)
        db.close()
    return 0 if total >= args.target else 2

if __name__ == '__main__':
    raise SystemExit(main())
