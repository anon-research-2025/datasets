using CSV, DataFrames
using Random
using StatsBase

const train_frac = 0.8
const val_frac = 0.1
const test_frac = 0.1

include("helpers.jl")

include("processing_subroutines.jl")


f = open("raw_data/deepTFBU/supp4.txt")

reads = read(f, String)
reads_split = split(reads, ">")
reads_split = split.(reads_split, '\n')
reads_split = reads_split[2:end]

reads_split_has_labels = filter(x->x[1][1][1]=='0', reads_split)
reads_split_has_labels |> length

headers = map(x -> x[1], reads_split_has_labels)
headers_split = split.(headers, '_')

labels = parse.(Float32, map(x -> x[1], headers_split))
seq = map(x -> x[2], reads_split_has_labels)

countmap(length.(seq))
selected_inds = length.(seq) .== 198
seq = seq[selected_inds]
labels = labels[selected_inds]
headers_split = headers_split[selected_inds]

seq = trim_common_ends(seq)
seq[1] |> length

@assert all(length.(seq) .== length(seq[1]))


tfs = ["HNF4A", "ELF1", "HNF1A"]

# HNF1A
selected_indices = tfs[1] .∈ headers_split
seq_HNF1A = seq[selected_indices]
labels_HNF1A = labels[selected_indices]

# ELF1
selected_indices = tfs[2] .∈ headers_split
seq_ELF1 = seq[selected_indices]
labels_ELF1 = labels[selected_indices]

# HNF4A
selected_indices = tfs[3] .∈ headers_split
seq_HNF4A = seq[selected_indices]
labels_HNF4A = labels[selected_indices]


using Plots
labels_HNF1A |> histogram
labels_ELF1 |> histogram
labels_HNF4A |> histogram


df_HNF1A = DataFrame(seq=seq_HNF1A, expr=labels_HNF1A)
df_ELF1 = DataFrame(seq=seq_ELF1, expr=labels_ELF1)
df_HNF4A = DataFrame(seq=seq_HNF4A, expr=labels_HNF4A)


for (df, n) in zip([df_HNF1A, df_ELF1, df_HNF4A], ["HNF1A", "ELF1", "HNF4A"])

    df_train, df_val, df_test = 
        process_this_df(df; expr_col_name="expr", _shuffle_=true)
    
    save_fasta_expr(df, df_train, df_val, df_test, 
        joinpath("deepTFBU", n),
            seq_col_name="seq", expr_col_name="expr")

    save_as_hdf5(
        df_train, df_val, df_test, n;
        folder_path="processed_data_tanh/deepTFBU/$n", 
        df_seq_col_name="seq",
        )
end



