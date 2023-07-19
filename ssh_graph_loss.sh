#!/bin/bash
rmz stats/t_losses.pt
rmz stats/v_losses.pt
scp ash1:~/g/vit/stats/t_losses.pt ./stats/t_losses.pt
scp ash1:~/g/vit/stats/v_losses.pt ./stats/v_losses.pt
python graph_loss.py
rmz stats/t_losses.pt
rmz stats/v_losses.pt
