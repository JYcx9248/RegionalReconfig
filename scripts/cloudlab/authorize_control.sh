#!/usr/bin/env bash
# Lets the control node (node0 of scripts/cloudlab/profile.py) log in to the hosts without a
# prompt, as run_cluster.py does (BatchMode). Run it from a machine that logs in to every node
# with a key of your CloudLab account, which the portal installs on each node: it makes a key on
# the control node if that has none, appends its public half to each host's authorized_keys,
# then logs in from the control node to each host's LAN address once, recording its host key.
# The private key never leaves the control node.
#
#   bash scripts/cloudlab/authorize_control.sh USER CONTROL HOST1 ... HOSTN
#
# CONTROL and the hosts are the nodes' public host names (the experiment's List View): node0,
# then node1 to nodeN in order, since nodeI is 10.10.1.I+1 on the LAN. The remote commands go
# through sh, whatever the account's login shell is.
set -euo pipefail
[ $# -ge 3 ] || { sed -n '2,13p' "$0" >&2; exit 2; }
user=$1 control=$2
shift 2

ssh "$user@$control" sh -s <<'EOF'
test -f ~/.ssh/id_ed25519 || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519
EOF
pub=$(ssh "$user@$control" cat .ssh/id_ed25519.pub)
case $pub in ssh-ed25519\ *) ;; *) echo "unexpected public key from $control: $pub" >&2; exit 1 ;; esac

for h in "$@"; do
  ssh "$user@$h" "sh -s -- '$pub'" <<'EOF'
mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
grep -qxF "$1" ~/.ssh/authorized_keys || printf '%s\n' "$1" >> ~/.ssh/authorized_keys
EOF
  echo "$h: the control node's key is authorized"
done

ssh "$user@$control" "sh -s -- $#" <<'EOF'
i=2
while [ $i -le $(($1 + 1)) ]; do
  a=10.10.1.$i
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$a" true; then
    echo "$a: the control node logs in"
  else
    echo "$a: the control node cannot log in" >&2
  fi
  i=$((i + 1))
done
EOF
