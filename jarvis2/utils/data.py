from sklearn import model_selection


def get_kfold_and_groups(a, data):
    if a.group_hashed:
        kfold = model_selection.GroupKFold(n_splits=a.n_splits)
        if a.group_name == "cv":
            groups = [get_candidate_hash(d) for d in data]
        elif a.group_name == "job":
            groups = [get_job_hash(d) for d in data]
        else:
            raise NotImplementedError
    else:
        kfold = model_selection.KFold(n_splits=a.n_splits)
        groups = None
    return kfold, groups


def get_candidate_hash(d):
    return hash(d["cv"]["jobtitle"] + "".join(sorted(d["cv"]["skills"])))


def get_cv_hash(d):
    return hash("".join(sorted(d["cv"]["skills"])))


def get_job_hash(d, use_jobtitle=False):
    return hash("".join(sorted(d["job"]["skills"])) + (d["job"]["jobtitle"] if use_jobtitle else ""))


def get_application_hash(a):
    return hash("".join(sorted(a["cv"]["skills"])) + a["job"]["jobtitle"] + "".join(sorted(a["job"]["skills"])) + str(
        a["label"]))
