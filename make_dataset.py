"""Builds data/train.csv, data/valid.csv, data/test.csv: a public stand-in for business documents
labeled marketing / legal / other.

    pip install datasets pandas scikit-learn
    python make_dataset.py

Sources and licenses are listed in README.md.
"""
import os
import re

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
MAX_CHARS = 3000          # long emails are cut; the notebooks read only the first ~384 tokens anyway
N_LEDGAR, N_TOS = 300, 200        # legal     = contract provisions + Terms-of-Service sentences
N_ADS, N_PROMO = 350, 150         # marketing = banner ad copy + promotional business emails
N_HAM = 500                       # other     = internal business emails


def clean(text):
    """Collapse whitespace, undo the 'word , word .' tokenization of the Enron and ToS texts, and lowercase.
    The Enron emails come lowercased only, so everything is lowercased: otherwise "has capitals" would
    give away "not other" and the model would learn letter case instead of content."""
    text = " ".join(text.lower().split())
    text = re.sub(r" ([,.!?;:%)\]])", r"\1", text)
    text = re.sub(r"([(\[$]) ", r"\1", text)
    text = re.sub(r" ' (s|t|re|ve|ll|d|m)\b", r"'\1", text)
    text = re.sub(r" - ", "-", text)
    return text[:MAX_CHARS]


# Promotional emails: Enron spam is mostly junk (pills, scams, stock pumps), so keep only
# business-style offers: a promo word + at least two business words + none of the junk words
PROMO = re.compile(r"\b(offer|discount|sale|save|savings|free shipping|special|deal|limited time|order now|"
                   r"buy now|shop now|coupon|catalog|promotion|best price|lowest price|free trial|sign up|guaranteed)\b")
BUSINESS = re.compile(r"\b(software|logo|website|web site|web design|design|marketing|advertising|seo|search engine|"
                      r"business|company|customers|clients|printing|toner|cartridges?|newsletter|conference|"
                      r"seminar|office|brand|corporate)\b")
JUNK = re.compile(r"\b(viagra|vlagra|cialis|xanax|valium|vicodin|ambien|pharm\w*|pills?|piils|meds|medi?cations?|"
                  r"medlcations|prescription|remedy|hiv|cancer|virus\w*|penis|enlarge\w*|sex\w*|porn|adult|xxx|nude|"
                  r"casino|gambl\w*|lottery|winner|inheritance|bank|password|verify|paypal|ebay|mortgage|refinanc\w*|"
                  r"loan|stocks?|investors?|invest|shares|shareholders|press release|otc|weight|diet|rolex|replica|"
                  r"watches|degree|diploma|bounce|non - member|dating|singles|tobacco|marlboro|cigarettes?|carton|oem|"
                  r"cheap|cd|dvd|warez|partnership|cooperation|beneficiary|funds|transfer|million|confidential|"
                  r"tablets?|taablets|chernist|chemist|rxdrugs|drug\w*|sa-ve|uregent|affiliate)\b")


def long_word_share(text):
    """Share of 14+ letter 'words': spam pads itself with glued nonsense like 'glassinealleyway'."""
    words = re.findall(r"[a-z]+", text.lower())
    return sum(len(w) > 13 for w in words) / max(len(words), 1)


def is_promo(text):
    return (PROMO.search(text) and len(BUSINESS.findall(text)) >= 2 and not JUNK.search(text)
            and long_word_share(text) <= 0.03)


rows = []

# legal
ledgar = load_dataset("coastalcph/lex_glue", "ledgar", split="train").shuffle(seed=SEED).select(range(N_LEDGAR))
rows += [{"text": clean(t), "label": "legal", "source": f"ledgar:{i}"} for i, t in enumerate(ledgar["text"])]
tos = load_dataset("coastalcph/lex_glue", "unfair_tos", split="train").shuffle(seed=SEED).select(range(N_TOS))
rows += [{"text": clean(t), "label": "legal", "source": f"unfair_tos:{i}"} for i, t in enumerate(tos["text"])]

# marketing
ads = load_dataset("PeterBrendan/Ads_Creative_Ad_Copy_Programmatic", split="train").to_pandas()
ads = ads[ads["text"].str.len() >= 30].drop_duplicates(subset="text").sample(N_ADS, random_state=SEED)
rows += [{"text": clean(t), "label": "marketing", "source": f"ads:{i}"} for i, t in zip(ads.index, ads["text"])]

enron = load_dataset("SetFit/enron_spam", split="train").to_pandas()
spam = enron[enron["label"] == 1].copy()
spam["body_start"] = spam["message"].fillna("").map(lambda m: " ".join(m.split())[:80])
spam = spam[spam["text"].map(lambda t: 200 <= len(t) <= 3000 and bool(is_promo(t)))]
spam = spam.drop_duplicates(subset="body_start").sample(N_PROMO, random_state=SEED)   # drop template copies
rows += [{"text": clean(t), "label": "marketing", "source": f"enron:{i}"} for i, t in zip(spam["message_id"], spam["text"])]

# other
ham = enron[(enron["label"] == 0) & (enron["text"].str.len() >= 80)].sample(N_HAM, random_state=SEED)
rows += [{"text": clean(t), "label": "other", "source": f"enron:{i}"} for i, t in zip(ham["message_id"], ham["text"])]

df = pd.DataFrame(rows).drop_duplicates(subset="text").sample(frac=1, random_state=SEED).reset_index(drop=True)

# Split 70 / 15 / 15, keeping the class balance the same in every part
train, rest = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=SEED)
valid, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=SEED)

os.makedirs("data", exist_ok=True)
for name, part in [("train", train), ("valid", valid), ("test", test)]:
    part.to_csv(f"data/{name}.csv", index=False)
    print(f"data/{name}.csv: {len(part)} documents, {part['label'].value_counts().to_dict()}")
