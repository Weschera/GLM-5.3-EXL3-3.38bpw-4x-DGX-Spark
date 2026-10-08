#!/usr/bin/env bash
# share-pack-nfs.sh - keep ONE copy of the 319 GB pack on one Spark and mount it read-only at the same path on the
# other three, over the ConnectX-7 fabric. Use this instead of `./glm53 setup --download-once` copying the pack to
# every Spark, when the Sparks do not each have ~330 GB free.
#
# Measured: our 84.4 Spark-Bench run loaded and served this way (loading over NFS is slower; decode is unaffected,
# weights are resident after load).
#
# usage (on the Spark that holds the pack, which does NOT have to be rank 0):
#   PACK=/home/$USER/models/GLM-5.3-EXL3-3.38bpw SUBNET=<fabric-subnet, e.g. 192.168.100.0/24> SERVER_IP=<this Spark's fabric IP> \
#     bash share-pack-nfs.sh <fabric-ip-1> <fabric-ip-2> <fabric-ip-3>
# Needs passwordless sudo on all four for nfs-kernel-server / mount (or run the printed lines yourself).
set -euo pipefail
: "${PACK:?set PACK}"; : "${SUBNET:?set SUBNET, e.g. 192.168.100.0/24}"; : "${SERVER_IP:?set SERVER_IP (fabric IP of this Spark)}"
[[ $# -eq 3 ]] || { echo "give the fabric IPs of the other three Sparks"; exit 1; }
dpkg -s nfs-kernel-server >/dev/null 2>&1 || sudo apt-get install -y nfs-kernel-server
sudo systemctl enable --now nfs-server
grep -q "^$PACK " /etc/exports || echo "$PACK $SUBNET(ro,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports >/dev/null
sudo exportfs -ra
for h in "$@"; do
  ssh -o BatchMode=yes "$h" "dpkg -s nfs-common >/dev/null 2>&1 || sudo apt-get install -y nfs-common; mkdir -p '$PACK'; \
    mountpoint -q '$PACK' || sudo mount -t nfs -o ro,vers=4.2,nconnect=8 $SERVER_IP:'$PACK' '$PACK'; \
    echo \"\$(hostname): \$(ls '$PACK'/*.safetensors | wc -l) shards visible\""
done
echo "Then: ./glm53 setup --model-dir $PACK   (same absolute path on all four)"
