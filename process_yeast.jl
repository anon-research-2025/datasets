
using HDF5, Random, StatsBase, DataFrames, CSV

data_where = "/home/shane/Desktop/seq2exp/seq2exp_datasets/raw_data/yeast/20160503_average_promoter_ELs_per_seq_atLeast100Counts.txt"
df = CSV.read(data_where, DataFrame; header=false)
rename!(df, :Column1 => :seq, :Column2 => :expr)

function longest_common_prefix(sequences)
    prefix = sequences[1]
    for seq in sequences[2:end]
        n = findfirst(i -> prefix[i] ≠ seq[i], 1:min(length(prefix), length(seq)))
        prefix = isnothing(n) ? prefix : prefix[1:n-1]
    end
    return prefix
end

function longest_common_suffix(sequences)
    reversed_seqs = [reverse(seq) for seq in sequences]
    suffix = longest_common_prefix(reversed_seqs)
    return reverse(suffix)
end

function trim_common_ends(sequences)
    prefix = longest_common_prefix(sequences)
    suffix = longest_common_suffix(sequences)
    plen, slen = length(prefix), length(suffix)
    return [seq[plen+1:end-slen] for seq in sequences]
end


countmap(length.(df.seq))

longest_common_prefix(df.seq)
longest_common_suffix(df.seq)

df.seq = trim_common_ends(df.seq)

countmap(length.(df.seq))

df = df[length.(df.seq).==84, :]



df_train, df_val, df_test = make_train_val_test_split(df)
df_train_max_expr = quantile(df_train[!,expr_col_name], quantile_max)
df_train_min_expr = quantile(df_train[!,expr_col_name], quantile_min)
max_minus_min = (df_train_max_expr-df_train_min_expr)

df_train_clipped = 
    minmax_clip.(df_train[!,expr_col_name], df_train_min_expr, df_train_max_expr)
df_val_expr_clipped = 
    minmax_clip.(df_val[!,expr_col_name], df_train_min_expr, df_train_max_expr)
df_test_expr_clipped = 
    minmax_clip.(df_test[!,expr_col_name], df_train_min_expr, df_train_max_expr)

df_train[!,"expr_minmax"] = 
    minmax_to_m1to1.((df_train_clipped .- df_train_min_expr) ./ max_minus_min)
df_val[!,"expr_minmax"] = 
    minmax_to_m1to1.((df_val_expr_clipped .- df_train_min_expr) ./ max_minus_min)
df_test[!,"expr_minmax"] =
    minmax_to_m1to1.((df_test_expr_clipped .- df_train_min_expr) ./ max_minus_min)

source_name = "Aviv"; this_data_set_name = "yeast"


save_fasta_expr(df, df_train, df_val, df_test, 
    joinpath(source_name, this_data_set_name);
    seq_col_name=seq_col_name, expr_col_name=expr_col_name)


fid = h5open("processed_data_tanh/Aviv/yeast/yeast.hdf5", "w")

col_name = "seq"

dna_array = Array{float_type, 3}(undef, (4, length(df_train[1,col_name]), size(df_train,1)))

for (ind,i) in enumerate(axes(df_train, 1))
    dna_array[:, :, i] = dna2dummy(df_train[i,col_name])
    ind % 10000 == 0 && println(ind)
end

labels = Array{float_type, 2}(undef, (1, size(df_train,1)))
for (ind,i) in enumerate(axes(df_train, 1))
    labels[1, i] = float_type(df_train.expr_minmax[i])
    ind % 10000 == 0 && println(ind)
end

fid["seq_1/train"] = dna_array
fid["expr/train"] = labels

dna_array = Array{float_type, 3}(undef, (4, length(df_val[1, col_name]), size(df_val,1)))
for (ind,i) in enumerate(axes(df_val, 1))
    dna_array[:, :, i] = dna2dummy(df_val[i,col_name])
    ind % 10000 == 0 && println(ind)
end

labels = Array{float_type, 2}(undef, (1, size(df_val,1)))
for (ind,i) in enumerate(axes(df_val, 1))
    labels[1, i] = float_type(df_val.expr_minmax[i])
    ind % 10000 == 0 && println(ind)
end

fid["seq_1/valid"] = dna_array
fid["expr/valid"] = labels

dna_array = Array{float_type, 3}(undef, (4, length(df_test[1,col_name]), size(df_test,1)))
for (ind,i) in enumerate(axes(df_test, 1))
    dna_array[:, :, i] = dna2dummy(df_test[i,col_name])
    ind % 10000 == 0 && println(ind)
end

labels = Array{float_type, 2}(undef, (1, size(df_test,1)))
for (ind,i) in enumerate(axes(df_test, 1))
    labels[1, i] = float_type(df_test.expr_minmax[i])
    ind % 10000 == 0 && println(ind)
end

fid["seq_1/test"] = dna_array
fid["expr/test"] = labels
close(fid)