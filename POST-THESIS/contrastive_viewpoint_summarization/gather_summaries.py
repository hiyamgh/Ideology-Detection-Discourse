import os
import pandas as pd


def process_summary_file(file_path, year):
    """
    Process a single file to extract the word and its two summaries.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    word = lines[0].split("Summary for ")[1].split(" from ")[0].strip()
    year = file_path.split("_")[-1].replace(".txt", "")

    summary1 = []
    summary2 = []
    summary_switch = False  # Switch between summaries based on the header

    for line in lines[1:]:
        if line.strip().startswith("Summary for"):
            summary_switch = True
            continue
        if summary_switch:
            summary2.append(line.strip().split('\t')[1])
        else:
            summary1.append(line.strip().split('\t')[1])

    summary1 = [s.replace(f"_{year}", "") for s in summary1]
    summary2 = [s.replace(f"_{year}", "") for s in summary2]

    # Join each summary into a single string (optional: preserve original format)
    summary1_text = ", ".join(summary1)
    summary2_text = ", ".join(summary2)

    # print(summary1_text)

    return word + f"_{year}", summary1_text, summary2_text


if __name__ == '__main__':

    # Example usage
    rows = []
    file_directory = "contrastive_summaries/aubmindlab-bert-base-arabertv2-monthly/"  # Replace with the directory containing your files
    for category in ["political_parties", "political_ideologies_positions", "sects", "ethnicities", "countries_locations"]:
        rootdir = "../generate_bert_embeddings/entities/contrastive_summaries/"
        category_words = []
        for file in os.listdir(rootdir):
            category_name = file.replace(".txt", "")
            if category_name == category:
                with open(os.path.join(rootdir, file), "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        w = line.strip().replace("\n", "")
                        category_words.append(w)
                f.close()
                break

        for i, w in enumerate(category_words):

            for year in ['06', '07', '08', '09', '10', '11', '12']:

                file_path = os.path.join(file_directory, f"{category}_word_{i}_{year}.txt")
                word, summary1, summary2 = process_summary_file(file_path, year)
                rows.append([word, "An-Nahar Viewpoint", summary1])  # Row for summary 1
                rows.append([word, "As-Safir Viewpoint", summary2])  # Row for summary 2
                rows.append([None, None])  # Empty row

    output_file = "summaries.xlsx"
    # Create DataFrame
    df = pd.DataFrame(rows, columns=["Word", "Viewpoint", "Summary"])

    # Save to Excel
    df.to_excel(output_file, index=False, sheet_name="Summaries")
    print(f"Excel sheet saved to {output_file}")

