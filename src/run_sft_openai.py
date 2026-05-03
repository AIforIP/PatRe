import os
import argparse
from openai import OpenAI
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str)
    args = parser.parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url)
    with open(args.train_file, "rb") as f:
        train_upload = client.files.create(file=f, purpose="fine-tune")
    val_upload_id = None
    if args.val_file:
        with open(args.val_file, "rb") as vf:
            val_upload = client.files.create(file=vf, purpose="fine-tune")
            val_upload_id = val_upload.id
    job = client.fine_tuning.jobs.create(
        training_file=train_upload.id,
        validation_file=val_upload_id,
        model=args.base_model,
    )
    print(job.id)
if __name__ == "__main__":
    main()


