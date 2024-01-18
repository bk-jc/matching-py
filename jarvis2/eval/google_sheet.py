import csv
import os


def compute_google_sheet(a, model, dataset):
    computed = []
    computed_bucket = []
    expected = []
    expected_bucket = []
    skills_cv = []
    skills_job = []

    for sample in dataset:
        computed_value = model(cv=[sample["cv"]], job=[sample["job"]]).detach().numpy()[0]
        computed.append(computed_value)
        computed_bucket.append("A" if computed_value > model.threshold else "C")
        expected.append(sample["label"])
        expected_bucket.append({1: "A", 0: "C"}[sample["label"]])
        skills_cv.append(len(sample["cv"]["skills"]))
        skills_job.append(len(sample["job"]["skills"]))

    gsheet_path = os.path.join(a.save_path, a.exp_name, a.version, "gsheet")
    os.makedirs(gsheet_path, exist_ok=True)

    # Attention: this seems to not work as intended. Importing the functions makes it such that the variables of the
    #  current sheet are not recognised. On the contrary, leaving them empty and importing them into an existing
    #  template overwrites the current cells. It thus seems easier to copy-paste the columns into the sheet manually.
    # TODO make a decision about this and remove the code.
    if False:
        columns = [
            ["" for _ in dataset],  # Table
            computed,  # Computed
            expected,  # Expected
            # [f'=IF(B{i + 1} > mean_sim,"A", IF(B{i + 1} > mean_disim, "B", "C"))'
            #  for i in range(len(dataset))],
            ["" for _ in dataset],  # Computed bucket
            # [f'=IF(C{i + 1} = 1, "A", IF(C{i + 1} = 0.5, "B", "C"))'
            #  for i in range(len(dataset))],
            ["" for _ in dataset],  # Expected bucket
            ["" for _ in dataset],  # Application ID
            ["" for _ in dataset],  # Job ID
            ["" for _ in dataset],  # Source platform
            skills_cv,  # Skills CV
            skills_job,  # Skills Job
            skills_cv,  # Native skills CV
            skills_job,  # Native skills Job
        ]

        rows = list(zip(*columns))
    
        with open(os.path.join(gsheet_path, 'import_sheet.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

    def save_list_to_file(data_list, filename):
        file_path = os.path.join(gsheet_path, filename)
        with open(file_path, 'w') as file:
            for item in data_list:
                file.write(f"{item}\n")

    save_list_to_file(computed_bucket, "computed_bucket.txt")
    save_list_to_file(computed, "computed.txt")
    save_list_to_file(expected_bucket, "expected_bucket.txt")
    save_list_to_file(expected, "expected.txt")
    save_list_to_file(skills_cv, "skills_cv.txt")
    save_list_to_file(skills_job, "skills_job.txt")
    save_list_to_file([model.threshold], "model_threshold.txt")
