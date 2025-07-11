import json
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
filename    = os.path.join(project_dir, "json_data", "combined_dataset.jsonl")
output      = os.path.join(project_dir, "json_data", "combined_dataset_cleaned.jsonl")

THRESHOLD = 0.81  # soglia per obs_ee[2]

def main():
    total_episodes   = 0
    kept_episodes    = 0
    removed_episodes = 0

    total_line       = 0   # conto di tutte le linee lette
    total_obs        = 0   # conto di tutti i data visti
    kept_obs         = 0   # conto di tutti i data mantenuti

    episode_buffer   = []  # accumula dict del singolo episodio
    episode_obs_cnt  = 0   # numero di data nel buffer

    removed_episode_idxs = []

    with open(filename, 'r') as f, open(output, 'w') as out_f:
        for line in f:
            total_line += 1
            entry = json.loads(line)
            episode_buffer.append(entry)

            if entry.get("type") == "data":
                total_obs     += 1
                episode_obs_cnt += 1

            if entry.get("type") == "episode_end":
                total_episodes += 1

                # Se ANY obs scende sotto soglia, scarta intero episodio
                if any(
                    e.get("type") == "data" and e["obs_ee"][2] < THRESHOLD
                    for e in episode_buffer
                ):
                    removed_episodes    += 1
                    removed_episode_idxs.append(total_line)
                    # non scrivo nulla, resetto buffer
                else:
                    # Scrivo tutti i data accumulati
                    for e in episode_buffer:
                        if e.get("type") == "data":
                            out_f.write(json.dumps(e) + "\n")
                    # Aggiorno kept_obs
                    kept_obs       += episode_obs_cnt
                    kept_episodes  += 1
                    # Scrivo il nuovo marker con indice corretto
                    end_marker = {
                        "type": "episode_end",
                        "episode_idx": kept_obs
                    }
                    out_f.write(json.dumps(end_marker) + "\n")

                # Reset buffer per il prossimo episodio
                episode_buffer  = []
                episode_obs_cnt = 0

    # Report finale
    print(f"Totale episodi letti:         {total_episodes}")
    print(f"Episodi mantenuti:            {kept_episodes}")
    print(f"Episodi rimossi:              {removed_episodes}")
    print(f"Totale osservazioni lette:    {total_obs}")
    print(f"Osservazioni mantenute:       {kept_obs}")
    print(f"Indici degli episodi rimossi: {removed_episode_idxs}")

if __name__ == "__main__":
    main()
