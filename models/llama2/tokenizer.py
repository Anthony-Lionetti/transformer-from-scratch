from huggingface_hub import hf_hub_download
from huggingface_hub import login
import sentencepiece as spm
from dotenv import load_dotenv
import os

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

tokenizer_file = hf_hub_download(
    repo_id="meta-llama/Llama-2-7b",
    filename="tokenizer.model",
    local_dir="Llama-2-7b"
)

class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)

    def decode(self, ids):
        return self.tokenizer.decode_pieces(ids)


tokenizer = LlamaTokenizer(tokenizer_file)