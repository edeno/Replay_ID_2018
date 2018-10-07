import pandas as pd


def _get_n_time_by_label(labels, overlap_labels):
    is_overlap_labels = labels.isin(
        overlap_labels.index.get_level_values(labels.name))
    overlap_labels1 = labels.loc[is_overlap_labels]
    overlap_labels1 = (overlap_labels1
                       .groupby(overlap_labels1)
                       .agg(len)
                       .rename(f'total_{overlap_labels1.name}'))

    return overlap_labels1


def compare_overlap(labels1, labels2, info1, info2):
    labels1 = labels1.copy().rename('labels1')
    labels2 = labels2.copy().rename('labels2')
    is_overlap = (labels1 > 0) & (labels2 > 0)

    overlap_labels = pd.concat(
        (labels1.loc[is_overlap], labels2.loc[is_overlap]), axis=1)
    overlap_labels = (overlap_labels
                      .groupby(overlap_labels.columns.tolist())
                      .agg(len)
                      .sort_index()
                      .rename('n_overlap')
                      .to_frame())

    overlap_labels1 = _get_n_time_by_label(labels1, overlap_labels)
    overlap_labels2 = _get_n_time_by_label(labels2, overlap_labels)

    name1 = 'overlap_percentage1'
    name2 = 'overlap_percentage2'

    percentage_overlap = {
        name1: lambda df: 100 * df.n_overlap / df[overlap_labels1.name],
        name2: lambda df: 100 * df.n_overlap / df[overlap_labels2.name]
    }

    overlap_labels = (overlap_labels
                      .join(overlap_labels1)
                      .join(overlap_labels2)
                      .assign(**percentage_overlap))

    start_time1 = (info1
                   .loc[overlap_labels.index.get_level_values(0).values,
                        ['start_time', 'end_time']])
    start_time2 = (info2
                   .loc[overlap_labels.index.get_level_values(1).values,
                        ['start_time', 'end_time']])

    time_difference = start_time1.values - start_time2.values

    overlap_labels['start_time_difference'] = time_difference[:, 0]
    overlap_labels['end_time_difference'] = time_difference[:, 1]

    return overlap_labels
