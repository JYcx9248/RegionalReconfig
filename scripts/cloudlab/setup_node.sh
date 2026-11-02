#!/usr/bin/env bash
# Prepares a CloudLab node for rtier (the wiki page Running on CloudLab), on any node type:
#   - the libraries the binaries link: run_cluster.py copies only the binaries to a host;
#   - the data disk at the lab server's paths, /data-ssd/chenxu and /netsys/chenxu: stage
#     records and warmup snapshots name files by absolute path;
#   - with --control (node0, the control node), what the build needs too: only it builds;
#   - with --gpu (c240g5), the NVIDIA driver and CUDA 12 from NVIDIA's repository for this
#     release: the P100 (Pascal) needs the driver's proprietary modules, of a branch no newer
#     than R580, and CUDA 12 (CUDA 13 dropped Pascal). The control node gets the toolkit, a host
#     only the runtime the engine links (libcudart.so.12). The driver loads after a reboot.
#
#   bash setup_node.sh [--control] [--gpu] [--disk DEVICE] [--yes]
#
# Without --yes it only prints what it would do. The data disk is the node's largest SSD that
# holds nothing mounted (--disk names one), formatted whole as XFS: a warmup snapshot is then
# restored by reflink, without a second copy on disk. Without such a disk, an SSD that is the
# boot disk gets a file system on its free space from CloudLab's mkextrafs (ext4, which has no
# reflink: a restore copies). Never a spinning disk: the nodes read raw vectors at random, page
# by page. The mounts are written to /etc/fstab (nofail), so a reboot keeps them.
set -euo pipefail

control=0 gpu=0 yes=0 disk=""
while [ $# -gt 0 ]; do
  case $1 in
    --control) control=1 ;;
    --gpu) gpu=1 ;;
    --yes) yes=1 ;;
    --disk) disk=${2:?--disk needs a device}; shift ;;
    *) sed -n '2,19p' "$0" >&2; exit 2 ;;
  esac
  shift
done
me=$(id -un)
data=/mnt/data

run() {
  if [ $yes = 1 ]; then echo "+ $*"; "$@"; else echo "would run: $*"; fi
}
# fstab LINE: appended unless an entry for its mount point is there.
fstab() {
  local mp
  mp=$(echo "$1" | awk '{print $2}')
  if awk -v mp="$mp" '$1 !~ /^#/ && $2 == mp {found = 1} END {exit !found}' /etc/fstab; then
    return
  fi
  if [ $yes = 1 ]; then echo "+ /etc/fstab: $1"; echo "$1" | sudo tee -a /etc/fstab >/dev/null
  else echo "would add to /etc/fstab: $1"; fi
}
mounts_of() { lsblk -n -o MOUNTPOINTS "$1" | grep . || true; }

# The largest non-rotational disk of 100 GB or more with nothing mounted; else the boot disk, if
# it is one.
pick_disk() {
  local name rota type size best="" bestsize=0 boot=""
  while read -r name rota type size; do
    case $name in zram* | ram* | loop*) continue ;; esac
    [ "$type" = disk ] && [ "$rota" = 0 ] && [ "$size" -ge 100000000000 ] || continue
    if [ -n "$(mounts_of "/dev/$name")" ]; then
      mounts_of "/dev/$name" | grep -qx / && boot=/dev/$name
    elif [ "$size" -gt "$bestsize" ]; then
      best=/dev/$name bestsize=$size
    fi
  done < <(lsblk -dn -b -o NAME,ROTA,TYPE,SIZE)
  echo "${best:-$boot}"
}

echo "== packages"
pkgs="rsync liburing2 libgomp1 zstd xfsprogs"
if [ $control = 1 ]; then
  pkgs="$pkgs build-essential cmake git liburing-dev python3-numpy tmux"
  # Go 1.24 or newer, from the release's own packages where it has them (Ubuntu 22.04, 24.04):
  # the Makefile sets GOTOOLCHAIN=local, so the build never fetches one.
  if apt-cache show golang-1.24 >/dev/null 2>&1; then pkgs="$pkgs golang-1.24"; go124=1; else go124=0; fi
fi
run sudo apt-get update -qq
# shellcheck disable=SC2086
run sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $pkgs
if [ $control = 1 ]; then
  if [ $go124 = 1 ]; then
    # A link on PATH whatever the login shell is (CloudLab accounts may use tcsh).
    run sudo ln -sf /usr/lib/go-1.24/bin/go /usr/lib/go-1.24/bin/gofmt /usr/local/bin/
  else
    echo "no golang-1.24 package on $(. /etc/os-release; echo "$PRETTY_NAME"): install Go 1.24 or newer" \
      "from https://go.dev/dl/ yourself (as on the wiki page Running on AWS, section 3)" >&2
  fi
fi
if [ $gpu = 1 ]; then
  echo "== NVIDIA driver and CUDA"
  cuda=12-8 driver=570   # 570 carries CUDA 12.8, the lab server's nvcc
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(. /etc/os-release; echo "${VERSION_ID/./}")/x86_64
    run wget -q -O /tmp/cuda-keyring.deb "$repo/cuda-keyring_1.1-1_all.deb"
    run sudo dpkg -i /tmp/cuda-keyring.deb
    run sudo apt-get update -qq
  fi
  # cuda-drivers-N is the proprietary flavor (nvidia-open has no Pascal support); DKMS builds it
  # against the running kernel's headers.
  gpkgs="linux-headers-$(uname -r) cuda-drivers-$driver"
  if [ $control = 1 ]; then gpkgs="$gpkgs cuda-toolkit-$cuda"; else gpkgs="$gpkgs cuda-cudart-$cuda"; fi
  run sudo env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $gpkgs
fi

echo "== data disk"
if mountpoint -q $data; then
  echo "$data is mounted already: $(findmnt -n -o SOURCE,FSTYPE $data)"
else
  [ -n "$disk" ] || disk=$(pick_disk)
  if [ -z "$disk" ]; then
    echo "no SSD of 100 GB or more found here:" >&2
    lsblk -d -o NAME,ROTA,TYPE,SIZE,MODEL >&2
    echo "name the data disk with --disk" >&2
    exit 1
  fi
  [ "$(lsblk -dn -o ROTA "$disk")" = 0 ] || echo "warning: $disk is a spinning disk: random reads will be slow" >&2
  lsblk -o NAME,ROTA,SIZE,FSTYPE,MOUNTPOINTS,MODEL "$disk"
  mounted=$(mounts_of "$disk")
  if [ -z "$mounted" ]; then
    echo "$disk holds nothing mounted: XFS on the whole disk (what is on it is lost)"
    run sudo wipefs -a "$disk"
    run sudo mkfs.xfs -f -q "$disk"
    run sudo mkdir -p $data
    run sudo mount "$disk" $data
    if [ $yes = 1 ]; then fstab "UUID=$(sudo blkid -s UUID -o value "$disk") $data xfs defaults,nofail 0 2"
    else echo "would add to /etc/fstab: UUID=<of $disk> $data xfs defaults,nofail 0 2"; fi
  elif echo "$mounted" | grep -qx /; then
    echo "$disk is the boot disk: CloudLab's mkextrafs makes a file system on its free space"
    if [ ! -x /usr/local/etc/emulab/mkextrafs.pl ]; then
      echo "no /usr/local/etc/emulab/mkextrafs.pl here: make a file system on the free space and mount it at $data" >&2
      exit 1
    fi
    run sudo mkdir -p $data    # mkextrafs wants its mount point to exist
    # -f: on a GPT boot disk (rs620) mkextrafs adds partition 4, then refuses it for its type
    # ("non-zero partition type ... use -f to override"); the partition is its own, just made.
    run sudo /usr/local/etc/emulab/mkextrafs.pl -f $data
  else
    echo "$disk has file systems mounted ($(echo "$mounted" | tr '\n' ' ')) and is not the boot disk:" \
      "name the data disk with --disk" >&2
    exit 1
  fi
fi

echo "== the lab server's paths"
for d in data-ssd netsys; do
  run sudo mkdir -p $data/$d /$d
  mountpoint -q /$d || run sudo mount --bind $data/$d /$d
  fstab "$data/$d /$d none bind,nofail 0 0"
  run sudo mkdir -p /$d/chenxu
  run sudo chown "$me:" /$d/chenxu
done

echo "== now"
df -h $data /data-ssd /netsys 2>/dev/null || true
if [ $control = 1 ] && [ $yes = 1 ]; then
  go version || true
  cmake --version | head -1
fi
if [ $gpu = 1 ] && [ $yes = 1 ]; then
  ldconfig -p | grep -q 'libcudart\.so\.12 ' || echo "warning: the loader does not find libcudart.so.12" >&2
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null ||
    echo "the NVIDIA driver loads after a reboot: sudo reboot, then nvidia-smi"
fi
[ $yes = 1 ] || echo "(nothing done: run again with --yes)"
