!#/bin/bash
rmz t_losses.pt
rmz v_losses.pt
scp ash1:~/g/vit/t_losses.pt ./t_losses.pt
scp ash1:~/g/vit/v_losses.pt ./v_losses.pt
python graph_loss.py
rmz t_losses.pt
rmz v_losses.pt