import json
import sys
import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))

sys.path.insert(0, REPO_ROOT)

try:
    from traffic_ppo import ActorCritic
except Exception:
    ActorCritic = None

MODEL_PATH = os.environ.get('PPO_MODEL_PATH', os.path.join(REPO_ROOT, 'ppo_model.pth'))


def load_model():
    if ActorCritic is None:
        return None
    if not os.path.exists(MODEL_PATH):
        return None
    model = ActorCritic(state_size=17, action_size=2)
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


model = load_model()

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
        req_id = req.get('id')
        state = req.get('state')
        if model is None or state is None:
            action = 0
        else:
            with torch.no_grad():
                tensor = torch.tensor([state], dtype=torch.float32)
                logits, _ = model(tensor)
                action = int(torch.argmax(logits, dim=-1).item())
        res = {'id': req_id, 'action': action}
        sys.stdout.write(json.dumps(res) + '\n')
        sys.stdout.flush()
    except Exception:
        res = {'id': None, 'action': 0}
        sys.stdout.write(json.dumps(res) + '\n')
        sys.stdout.flush()
