#!/bin/bash
# GLM-5.3 EXL3 3.38bpw (vcruz305, SAGE MixedK) on all four Sparks via Vic's recipe (ed2f31d, TensorFold TP4 + DFlash2).
# Deviations from DEPLOY.md, all forced by our cluster: hosts file written by hand (all four Sparks share one cloned
# /etc/machine-id, so init refuses); pack lives on <spark> and is NFS-mounted read-only at the same path on the
# other three (not enough disk for four 319 GB copies); /v1 reached from the bench client through an ssh tunnel
# (recipe default loopback bind kept). Profile fast-160k (the measured default), thinking off, temp 1.0 / top_p 0.95.
set -u
OUT=~/projects/lg-hard-pilot/rec; LOG=$OUT/QUEUE.log
H0=<user>@<spark>; HB=<user>@<spark>
PACK=/home/<user>/models/GLM-5.3-EXL3-3.38bpw; REV=cc64e77b1b1744c7bc71e54c838727f0db0f1a36
say(){ echo "[$(date '+%H:%M')] g338: $*" >> $LOG; }
die(){ say "STOPPED: $*"; exit 1; }

if [ "${RESUME_EXPORT:-0}" != 1 ]; then
say "re-verifying b610 checkpoints against NAS (fixed manifest bug: NAS side listed 0 files)"
scp -q ~/projects/lg-hard-pilot/g338/bvd.sh $HB:~/bvd.sh
A=<nas-archive>
ssh $HB "rm -f ~/bvd-*.log; nohup bash ~/bvd.sh ~/models/nvidia-GLM-5.3-Flash-NVFP4 <nas-archive>/<spark>/nvidia-GLM-5.3-Flash-NVFP4 verify >/dev/null 2>&1 </dev/null & nohup bash ~/bvd.sh ~/models/DeepSeek-V4.1-Flash-EXL3-2.9bpw <nas-archive>/<spark>/DeepSeek-V4.1-Flash-EXL3-2.9bpw verify >/dev/null 2>&1 </dev/null &" </dev/null
sleep 20
say "waiting for b610 backups to verify before freeing its disk"
until ssh $HB 'n=0; for f in ~/bvd-*.log; do grep -q RESULT $f && n=$((n+1)); done; [ $n -ge 2 ]' </dev/null; do sleep 60; done
ssh $HB 'grep -h RESULT ~/bvd-*.log' </dev/null >> $LOG
ssh $HB '[ $(grep -l "RESULT: MATCH" ~/bvd-*.log | wc -l) -ge 2 ]' </dev/null || die "b610 backup mismatch; kept its models, need a decision"

say "downloading pack @${REV:0:7} to <spark>"
ssh $HB "mkdir -p $PACK && ~/.local/bin/hf download vcruz305/GLM-5.3-EXL3-3.38bpw --revision $REV --local-dir $PACK --max-workers 8 > ~/g338-dl.log 2>&1" </dev/null \
  || die "pack download failed (see b610 ~/g338-dl.log)"
ssh $HB "cd $PACK && sha256sum config.json | cut -c1-64" </dev/null | grep -q f520ce2795fa4f38a835f84ad4d3db17983d63babb7bb16f6e23178b4612ba18 || die "pack config sha differs from recipe pin"
fi
if [ "${RESUME_SETUP:-0}" != 1 ]; then
say "pack on b610 ($(ssh $HB "du -sh $PACK | cut -f1" </dev/null)); exporting over NFS to the other three"
ssh $HB "sudo -n systemctl enable --now nfs-server >/dev/null 2>&1; grep -q '$PACK ' /etc/exports || echo '$PACK <fabric-subnet>(ro,no_subtree_check,no_root_squash)' | sudo -n tee -a /etc/exports >/dev/null; sudo -n exportfs -ra" </dev/null || die "NFS export failed"
for h in <spark-0> <spark-2> <spark-3>; do
  ssh <user>@$h "mkdir -p $PACK; mountpoint -q $PACK || sudo -n mount -t nfs -o ro,vers=4.2,nconnect=8 <pack-spark-fabric-ip>:$PACK $PACK; ls $PACK/*.safetensors | wc -l" </dev/null >> $LOG 2>&1 || die "NFS mount failed on $h"
done

fi
if [ "${RESUME_UP:-0}" != 1 ]; then
say "recipe setup on all four (runtime build + drafter + BF16 lm_head + serve view)"
ssh $H0 "cd ~/GLM-5.3-EXL3-DGX-Spark-recipe && ./glm53 setup --model-dir $PACK > ~/g338-setup.log 2>&1" </dev/null \
  || { ssh $H0 'tail -40 ~/g338-setup.log' </dev/null > $OUT/g338-setup-fail.log; die "setup failed (rec/g338-setup-fail.log)"; }
ssh $H0 'tail -6 ~/g338-setup.log' </dev/null >> $LOG

fi
say "waiting for the CYBER-FROST bench to finish so every GPU is free"
while pgrep -f run_cf387_off.sh >/dev/null; do sleep 60; done
ssh $H0 'pkill -f "exllamav3-tabby/serve.sh|tabbyAPI|qwen38-exl3/venv" ; sleep 10' </dev/null
for h in <spark-0> <spark-pack> <spark-2> <spark-3>; do ssh <user>@$h 'docker ps -q --filter name=rs249 | xargs -r docker rm -f >/dev/null' </dev/null; done

for h in <spark-0> <spark-pack> <spark-2> <spark-3>; do ssh <user>@$h 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sudo -n sysctl -q vm.swappiness=1; sudo -n swapoff -a; rm -f /tmp/tf_serve_memwatch.VIOLATION' </dev/null; done
say "swap off on all four for this run: 9f73 grew 4 kB of swap at 102 GiB free during load (twice, swappiness 1), which trips the recipe watchdog; its MemAvailable guard stays on"
say "./glm53 up (fast-160k)"
ssh $H0 "cd ~/GLM-5.3-EXL3-DGX-Spark-recipe && ./glm53 up > ~/g338-up.log 2>&1" </dev/null \
  || { ssh $H0 'tail -50 ~/g338-up.log' </dev/null > $OUT/g338-up-fail.log; die "up failed (rec/g338-up-fail.log)"; }
ssh $H0 'grep -E "preflight OK|READY" ~/g338-up.log' </dev/null >> $LOG
ssh $H0 "cd ~/GLM-5.3-EXL3-DGX-Spark-recipe && ./glm53 smoke > ~/g338-smoke.log 2>&1; tail -3 ~/g338-smoke.log" </dev/null >> $LOG

# tunnel: bench client -> rank 0 loopback
( while true; do ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 18890:127.0.0.1:8890 $H0; sleep 5; done ) &
TUN=$!; trap 'kill $TUN 2>/dev/null; pkill -f "18890:127.0.0.1:8890"' EXIT
for i in $(seq 1 20); do curl -s -m 5 http://127.0.0.1:18890/v1/models | grep -q GLM-5.3 && break; sleep 3; done
curl -s -m 5 http://127.0.0.1:18890/v1/models | grep -q GLM-5.3 || die "tunnel to /v1 not answering"
say "serving; starting bench"

cd ~/projects/spark-bench-development; COMMIT=$(git rev-parse --short HEAD)
export SPARK_BENCH_SEED_PER_REPEAT=2000 PYTHONUNBUFFERED=1 SPARK_BENCH_FORCE_TEMPERATURE=1.0
SPARK_BENCH_CONTEXT=163840 SPARK_BENCH_EXTRA_BODY='{"top_p":0.95}' ~/venvs/macbench-mlx/bin/python spark_bench.py eval \
  --label "GLM-5.3 EXL3 SAGE MixedK 3.38bpw (Vic recipe ed2f31d, TensorFold TP4 + DFlash2, 4 Sparks, fast-160k) thinking OFF" \
  --endpoint http://127.0.0.1:18890/v1 --model GLM-5.3-EXL3-3.38bpw --topology "4x DGX Spark TP4 (RoCE)" \
  --thinking off --uncapped --temperature 1.0 --repeats 2 --tier all --timeout 7200 --preflight-loose \
  --out-dir $OUT/g338-off --run-kind benchmark \
  --notes "swap disabled during run (kernel 4 kB swap-outs at 100+ GiB free tripped the watchdog); vcruz305/GLM-5.3-EXL3-3.38bpw @cc64e77 (pack pin; card ca06664); recipe ed2f31d: TensorFold 757a851, b12x b58f34e RoCE, DFlash2 incoai/GLM-5.3-DFlash2 @425aa61 d7 c0.6, BF16 lm_head view, fixed template; profile fast-160k; deviations: hand-written hosts (shared machine-id), pack NFS ro from one Spark, ssh tunnel; temp 1.0 top_p 0.95; UNCAPPED (window 163840); bench @$COMMIT; seeds 2001-2002" \
  > $OUT/g338-off.log 2>&1
say "g338-off exit=$? $(grep -o 'TrueScore [0-9.]*/100' $OUT/g338-off.log | tail -1)"
