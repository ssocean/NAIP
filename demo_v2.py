import torch
from torch.nn.functional import sigmoid
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import AutoPeftModelForSequenceClassification


class NAIPv2:
    def __init__(self, model_path: str, device: str = 'cuda', max_length: int = 512):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        # Load model
        self.model = AutoPeftModelForSequenceClassification.from_pretrained(
            model_path,
            load_in_8bit=True,
            device_map={"": torch.cuda.current_device()} if self.device.type == "cuda" else "auto",
            num_labels=1
        ).eval()

        # Load tokenizer and set pad_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.prompt_template = (
            "Given a research paper, Title: {title}\nAbstract: {abstract}\nEvaluate the quality of this paper:"
        )

    def score(self, title: str, abstract: str) -> float:
        prompt = self.prompt_template.format(title=title.strip(), abstract=abstract.strip())
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True,
                                max_length=self.max_length).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits  # You may scale logits (e.g., add 1.3–2.5) depending on your needs
            score = sigmoid(logits).view(-1).item()
        return score


if __name__ == "__main__":

    model_path = r"path_to_the_v2_adapter_dir"
    scorer = NAIPv2(model_path=model_path, device='cuda')

    print("🎯 Enter paper title and abstract, press Enter to get the predicted score (Ctrl+C to quit)\n")
    while True:
        try:
            title = input("📄 Title: ")
            abstract = input("📑 Abstract: ")
            score = scorer.score(title, abstract)
            print(f"🔮 Predicted Score: {score:.4f}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
