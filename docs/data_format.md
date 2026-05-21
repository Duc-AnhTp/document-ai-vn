# Donut data format

Du lieu processed cho MC-OCR nam tai:

```text
data/mc-ocr/donut_format/
  train/
    metadata.jsonl
    *.jpg
  val/
    metadata.jsonl
    *.jpg
  test/
    metadata.jsonl
    *.jpg
```

## `metadata.jsonl`

Moi dong la mot JSON object:

```json
{
  "file_name": "mcocr_public_145013aagqw.jpg",
  "ground_truth": "{\"gt_parse\": {\"store_name\": \"CUA HANG ABC\", \"date\": \"01/01/2021\", \"total\": \"150000\", \"address\": \"123 Nguyen Hue\"}}"
}
```

`ground_truth` la chuoi JSON long, khong phai object truc tiep. Khi load can `json.loads(record["ground_truth"])` de lay `gt_parse`.

## Schema KIE

| Field | Y nghia |
| --- | --- |
| `store_name` | Ten cua hang hoac don vi ban hang |
| `date` | Ngay tren bien lai |
| `total` | Tong tien |
| `address` | Dia chi cua hang |

Tat ca record giu du 4 key. Neu field khong co annotation, value la chuoi rong `""`.

## Rebuild data

```powershell
python scripts/download_data.py --dataset mcocr --output data/
python scripts/convert_mcocr.py --input data/mc-ocr/raw/ --output data/mc-ocr/donut_format/ --split-ratio 0.8 0.1 0.1
```

Mac dinh converter dung seed `42`, split `0.8/0.1/0.1`, normalize Unicode NFC va copy anh vao tung split.
