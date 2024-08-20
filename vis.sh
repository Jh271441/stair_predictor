#!/bin/bash

input_dir="labelme_json_3"
output_dir="labelme_json_viz_3"

mkdir -p "$output_dir"

for json_file in "$input_dir"/*.json; do
#  echo ${json_file:13:20}
  labelme_export_json "$json_file" -o "$output_dir/${json_file:15:30}"
  echo "Processed $json_file"
done

