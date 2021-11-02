#!/usr/bin/env python3
# Run from root of project

"""
## Downloading Sent2vec Pre-Trained Models (https://github.com/epfml/sent2vec/)

- [sent2vec_wiki_unigrams](https://drive.google.com/open?id=0B6VhzidiLvjSa19uYWlLUEkzX3c) 5GB (600dim, trained on english wikipedia)
- [sent2vec_wiki_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSaER5YkJUdWdPWU0) 16GB (700dim, trained on english wikipedia)
- [sent2vec_twitter_unigrams](https://drive.google.com/open?id=0B6VhzidiLvjSaVFLM0xJNk9DTzg) 13GB (700dim, trained on english tweets)
- [sent2vec_twitter_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSeHI4cmdQdXpTRHc) 23GB (700dim, trained on english tweets)
- [sent2vec_toronto books_unigrams](https://drive.google.com/open?id=0B6VhzidiLvjSOWdGM0tOX1lUNEk) 2GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))
- [sent2vec_toronto books_bigrams](https://drive.google.com/open?id=0B6VhzidiLvjSdENLSEhrdWprQ0k) 7GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))
"""
import os

import gdown

from utils import project_path, ensure_dir

save_dir = os.path.join(project_path, "saved/s2v")
ensure_dir(save_dir)

model_name_to_url = {
    "wiki_unigrams": "https://drive.google.com/uc?id=0B6VhzidiLvjSa19uYWlLUEkzX3c",
    # 5GB (600dim, trained on english wikipedia)
    "wiki_bigrams": "https://drive.google.com/uc?id=0B6VhzidiLvjSaER5YkJUdWdPWU0",
    # 16GB (700dim, trained on english wikipedia)
    "twitter_unigrams": "https://drive.google.com/uc?id=0B6VhzidiLvjSaVFLM0xJNk9DTzg",
    # 13GB (700dim, trained on english tweets)
    "twitter_bigrams": "https://drive.google.com/uc?id=0B6VhzidiLvjSeHI4cmdQdXpTRHc",
    # 23GB (700dim, trained on english tweets)
    "torontobooks_unigrams": "https://drive.google.com/uc?id=0B6VhzidiLvjSOWdGM0tOX1lUNEk",
    # 2GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))
    "torontobooks_bigrams": "https://drive.google.com/uc?id=0B6VhzidiLvjSdENLSEhrdWprQ0k"
    # 7GB (700dim, trained on the [BookCorpus dataset](http://yknzhu.wixsite.com/mbweb))
}

if __name__ == '__main__':
    names_of_models_to_download = "twitter_unigrams", "twitter_bigrams"
    for model_name in names_of_models_to_download:
        gdown.download(model_name_to_url[model_name], os.path.join(save_dir, f"{model_name}.bin"), quiet=False)
