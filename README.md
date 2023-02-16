# GPT-from-scratch
This repository is a re-implementation of Andrej Karpathy's nanoGPT. It is trained on the Tiny Shakespeare Dataset using Transformer Architecture.

## Dependencies
We only need PyTorch for this repository. I created requirements.txt to include the version I use and to add further possible dependencies that may arrive in future.
```bash
cd PROJECT_PATH
pip -r requirements.txt
```

## Getting Started
All the hyperparameters can be found in the body of train.py. After arranging them to your desire, you can call 

```bash
python train.py
```

and that's it! It takes around 15 minutes on a single RTX 3090.


## Generated Texts

I used the model that has the best validation score to generate some texts. Here are a few examples:

```text
First Citizen:
Thou drink me, prick thee with violence. The vest,
could noise the valour tears of my veins
sheelf captain and walked on her is to be
saily: I would it be it in not ask; that
prophest he shall reported to heaven
the heavens, whose hath only she well to poss. Even
he came to for my wanton audic testrate of myself:
Whom thou have strange force thyself tears:
it demand thee, thou calls, make too princies,
fear, if Oxford thou wert through it shall ere root.
```

```text
WICKING RICHARD III:
But unfold, gear me the child, I'll revenge
They and, in threw'd, and law must surfeils;
I knock'd, he's death, which they foreget,
Cuount the sunshines yours, church ambs;
Furst thence they duty use dews, us I grustly;
And then him purposses in sometitute,
To bide me down of their earth drown bold!
O wolt, I am sugg'st husbily sound and mistrest
Slungs. Warwick! King Richard, Margius lives.
```


## References

[1]	Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

[2] Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT)