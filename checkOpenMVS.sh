if ! [ -x "$(command -v DensifyPointCloud)" ]; then
  echo 'Error: OpenMVS/OpenMVG is not installed.' >&2
  exit 1
fi