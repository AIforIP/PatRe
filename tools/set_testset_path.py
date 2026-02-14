import argparse
import re


def main():
    parser = argparse.ArgumentParser(description="Update test_set path in config/eval_config.yaml")
    parser.add_argument("--config", default="config/eval_config.yaml", help="Config path")
    parser.add_argument("--test-set", required=True, help="New test_set path")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        text = f.read()

    pattern = r"(\btest_set\s*:\s*)(\"[^\"]*\"|'[^']*'|[^\n#]+)"
    repl = r"\1\"%s\"" % args.test_set.replace("\\", "/")

    if not re.search(pattern, text):
        raise SystemExit("Could not find test_set in config.")

    new_text = re.sub(pattern, repl, text, count=1)

    with open(args.config, "w", encoding="utf-8") as f:
        f.write(new_text)

    print(f"Updated {args.config}: test_set -> {args.test_set}")


if __name__ == "__main__":
    main()
