input_path = "/home/lar/Riccardo/diffusion_mk2/json_data/combined_dataset.jsonl"
output_path = "/home/lar/Riccardo/diffusion_mk2/json_data/combined_dataset_fixed.jsonl"

with open(input_path, "r") as infile:
    lines = infile.readlines()

initial_line_count = len(lines)
episode_end_count = 0
cleaned_lines = []
skip_next = False

for i, line in enumerate(lines):
    if i == 0:
        continue  # salta la prima riga del file

    if skip_next:
        skip_next = False
        continue  # salta la riga dopo "episode_end"

    cleaned_lines.append(line)

    if '"type": "episode_end"' in line:
        episode_end_count += 1
        skip_next = True

final_line_count = len(cleaned_lines)

with open(output_path, "w") as outfile:
    outfile.writelines(cleaned_lines)

# Stampa dei risultati
print(f"Numero righe iniziali: {initial_line_count}")
print(f"Numero episodi (episode_end): {episode_end_count}")
print(f"Numero righe finali: {final_line_count}")
print(f"Numero righe finali corrette: {initial_line_count-episode_end_count}")
