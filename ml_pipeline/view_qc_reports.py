#!/usr/bin/env python3
"""
Download DeepPrep QC reports from FABRIC VMs to local machine and
open them in a browser.

DeepPrep generates HTML QC reports for each subject that show:
  - T1 registration alignment
  - Skull stripping quality
  - Brain segmentation
  - BOLD-to-T1 coregistration
  - Motion trace

These are critical for validating preprocessing — bad registration = garbage FC.

Usage (run from inside JupyterHub notebook):
    from view_qc_reports import collect_qc_reports
    collect_qc_reports(slice, SUBJECT_NODE_MAP)
"""

import os
import subprocess
import webbrowser


def collect_qc_reports(slice_obj, subject_node_map,
                       output_dir='/home/fabric/work/qc_reports',
                       ssh_key='/home/fabric/work/fabric_config/slice_key',
                       ssh_config='/home/fabric/work/fabric_config/ssh_config'):
    """Copy QC HTML reports from each FABRIC node to local machine."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading DeepPrep QC reports to {output_dir}...\n")

    for sub, node_name in subject_node_map.items():
        node = slice_obj.get_node(node_name)
        ip = node.get_management_ip()

        # DeepPrep QC reports are in the output/qc directory
        remote_qc = f'/home/ubuntu/sudmex/deepprep_output/QC/sub-{sub}'
        local_qc = f'{output_dir}/sub-{sub}'
        os.makedirs(local_qc, exist_ok=True)

        cmd = (
            f'scp -r -o StrictHostKeyChecking=no '
            f'-F {ssh_config} -i {ssh_key} '
            f'ubuntu@\\[{ip}\\]:{remote_qc}/* {local_qc}/'
        )
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode == 0:
            html_files = [f for f in os.listdir(local_qc) if f.endswith('.html')]
            print(f"  ✅ sub-{sub}: {len(html_files)} QC HTML files")
        else:
            # Fallback: DeepPrep may use a different path
            alt_remote = f'/home/ubuntu/sudmex/deepprep_output/reports/sub-{sub}*'
            cmd_alt = (
                f'scp -r -o StrictHostKeyChecking=no '
                f'-F {ssh_config} -i {ssh_key} '
                f'ubuntu@\\[{ip}\\]:{alt_remote} {local_qc}/'
            )
            subprocess.run(cmd_alt, shell=True, capture_output=True, text=True)
            print(f"  ⚠️ sub-{sub}: tried fallback path — check {local_qc}")

    # Create a simple index.html that links to all subject reports
    index_path = f'{output_dir}/index.html'
    with open(index_path, 'w') as f:
        f.write("""<!DOCTYPE html>
<html><head><title>DeepPrep QC Reports</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; }
h1 { color: #e94560; }
ul { list-style: none; padding: 0; }
li { margin: 10px 0; padding: 15px; background: #f5f5f5; border-radius: 8px; }
a { color: #1e88e5; text-decoration: none; font-weight: bold; }
a:hover { text-decoration: underline; }
.note { background: #fff3cd; padding: 15px; border-radius: 8px; margin: 20px 0; }
</style></head>
<body>
<h1>DeepPrep Quality Control Reports</h1>
<div class="note">
Click each subject to inspect registration, skull stripping, and BOLD coregistration.
Poor quality in any step means the FC matrix for that subject is unreliable.
</div>
<ul>
""")
        for sub in sorted(subject_node_map.keys()):
            sub_dir = f'sub-{sub}'
            if os.path.exists(f'{output_dir}/{sub_dir}'):
                htmls = [f for f in os.listdir(f'{output_dir}/{sub_dir}') if f.endswith('.html')]
                if htmls:
                    f.write(f'<li>sub-{sub}: ')
                    for h in htmls:
                        f.write(f'<a href="{sub_dir}/{h}">{h}</a> &nbsp; ')
                    f.write('</li>\n')
                else:
                    f.write(f'<li>sub-{sub}: (no HTML reports found)</li>\n')
        f.write('</ul></body></html>')

    print(f"\n✅ Index: file://{index_path}")
    print(f"Open in browser: {index_path}")


def list_available_qc_paths(slice_obj, subject_node_map):
    """Check what QC outputs exist on each node. Useful for debugging."""
    print("Checking DeepPrep output structure on each node...\n")
    for sub, node_name in subject_node_map.items():
        node = slice_obj.get_node(node_name)
        print(f"=== {node_name} (sub-{sub}) ===")
        ls, _ = node.execute(f'find /home/ubuntu/sudmex/deepprep_output -name "*.html" -path "*sub-{sub}*" 2>/dev/null | head -20')
        if ls.strip():
            print(ls)
        else:
            # Broader search
            ls, _ = node.execute(f'find /home/ubuntu/sudmex/deepprep_output -name "*sub-{sub}*" 2>/dev/null | head -20')
            print(ls if ls.strip() else "(no files found)")
        print()


if __name__ == "__main__":
    print("This module should be imported from the JupyterHub notebook.")
    print("Usage:")
    print("  from view_qc_reports import collect_qc_reports")
    print("  collect_qc_reports(slice, {'001': 'DeepPrepNode1', ...})")
