import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PaperScorer:
    def __init__(self, model_path: str, device: str = "cuda", max_length: int = 512):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_length = max_length

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=1,
            load_in_8bit=True
        ).to(self.device).eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # ⚠️ DO NOT CHANGE PROMPT
        self.prompt_template = (
            f'''Given a certain paper, Title: {title}\n Abstract: {abstract}. \n Predict its normalized academic impact (between 0 and 1):'''
        )

    def score(self, title: str, abstract: str) -> float:
        """Return predicted impact score (between 0 and 1)."""
        text = self.prompt_template.format(
            title=title.strip().replace("\n", ""),
            abstract=abstract.strip().replace("\n", "")
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = torch.sigmoid(outputs["logits"]).item()

        return score


if __name__ == "__main__":
    model_path = r"path_to_the_v1_dir"
    scorer = PaperScorer(model_path=model_path, device="cuda")

    print("🎯 Enter paper title and abstract. Press Ctrl+C to quit.\n")

    while True:
        try:
            title = input("Enter a title: ")
            abstract = input("Enter an abstract: ")

            score = scorer.score(title, abstract)
            print(f"🔮 Predicted Impact Score: {score:.4f}\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break

