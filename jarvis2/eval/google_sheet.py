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
