def load_bert():
    from transformers import BertTokenizer, BertModel, BertConfig
    import transformers

    transformers.BertModel.from_pretrained('bert-base-uncased')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def train_sha_txt():
    import sentencepiece as spm

    spm.SentencePieceTrainer.Train(
        '--input=input.txt --model_prefix=sh ' +
        '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --pad_piece=<pad> ' +
        '--split_by_whitespace=false --model_type=unigram ' +
        '--user_defined_symbols=<sep>,<cls>,<mask> --vocab_size=200 --character_coverage=1')


if __name__ == '__main__':
    train_sha_txt()