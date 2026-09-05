"""Create a small, reproducible knowledge-graph completion dataset."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

TRAIN = """Beijing\tcapital_of\tChina
Shanghai\tlocated_in\tChina
Shenzhen\tlocated_in\tChina
Paris\tcapital_of\tFrance
Berlin\tcapital_of\tGermany
Tokyo\tcapital_of\tJapan
China\tlocated_in\tAsia
Japan\tlocated_in\tAsia
France\tlocated_in\tEurope
Germany\tlocated_in\tEurope
Beijing\tconnected_to\tShanghai
Paris\tconnected_to\tLyon
Berlin\tconnected_to\tMunich
Tokyo\tconnected_to\tOsaka
"""
VALID = """Shanghai\tconnected_to\tShenzhen
Shenzhen\tconnected_to\tBeijing
"""
TEST = """Lyon\tlocated_in\tFrance
Munich\tlocated_in\tGermany
Osaka\tlocated_in\tJapan
"""


def main() -> None:
    for name, content in [("train.tsv", TRAIN), ("valid.tsv", VALID), ("test.tsv", TEST)]:
        (ROOT / name).write_text(content, encoding="utf-8", newline="\n")
    print(f"wrote toy KG triples to {ROOT}")


if __name__ == "__main__":
    main()
