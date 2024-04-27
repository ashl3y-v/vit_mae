#!/bin/bash
rmz stats/test.pt stats/test_hat.pt
ssh ash1 -t "cd ~/g/vit/; python ~/g/vit/test.py 1"
scp ash1:~/g/vit/stats/test.pt ~/g/vit/stats/test.pt
scp ash1:~/g/vit/stats/test_hat.pt ~/g/vit/stats/test_hat.pt
python view.py ./stats/test.pt ./stats/test_hat.pt
