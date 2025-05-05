using CSV, DataFrames
using Random
using StatsBase

const train_frac = 0.8
const val_frac = 0.1
const test_frac = 0.1

const processed_data_path = "processed_data"
const raw_data_path = "raw_data"

const quantile_max = 0.99
const quantile_min = 0.01

gpro = "gpro"
train_set = "training_set"
val_set = "validation_set"
test_set = "test_set"

include("helpers.jl")

minmax_clip(val, min, max) = 
    val < min ? min : (val > max ? max : val)

minmax_to_m1to1(x) = 2*x - 1

function make_train_val_test_split(df)
    train_ind_end = floor(Int, train_frac*size(df, 1))
    val_ind_start = train_ind_end + 1
    val_ind_end = floor(Int, (train_frac+val_frac)*size(df, 1))
    test_ind_start = val_ind_end + 1
    test_ind_end = size(df, 1)
    @info "train_ind_end: $train_ind_end, val_ind_start: $val_ind_start, val_ind_end: $val_ind_end, test_ind_start: $test_ind_start, test_ind_end: $test_ind_end"
    df_train = df[1:train_ind_end, :]
    df_val = df[val_ind_start:val_ind_end, :]
    df_test = df[test_ind_start:test_ind_end, :]
    return df_train, df_val, df_test
end

# Define the function to write the FASTA file
function write_fasta(filename, sequences)
    open(filename, "w") do fasta_file
        for (i, seq) in enumerate(sequences)
            println(fasta_file, ">sequence_$i")
            println(fasta_file, seq)
        end
    end
end
function write_real_values(filename, values)
    open(filename, "w") do file
        for value in values
            println(file, value)
        end
    end
end

function save_fasta_expr(df, df_train, df_val, df_test, this_data_set_name, 
    ;seq_col_name="seq", expr_col_name="expr", 
    expr_minmax_col_name="expr_minmax",
    save_path="processed_data_tanh", train_set="training_set", 
    val_set="validation_set", test_set="test_set")

    mkpath(joinpath(save_path, this_data_set_name))
    CSV.write(joinpath(save_path, this_data_set_name, "data.csv"), df)

    save_where = joinpath(save_path, this_data_set_name, train_set)
    mkpath(save_where)
    write_fasta("$save_where/seqs.fa", df_train[!, seq_col_name])
    write_real_values("$save_where/expr.txt", df_train[!, expr_col_name])
    write_real_values("$save_where/expr_minmax.txt", df_train[!, expr_minmax_col_name])

    save_where = joinpath(save_path, this_data_set_name, val_set)
    mkpath(save_where)
    write_fasta("$save_where/seqs.fa", df_val[!, seq_col_name])
    write_real_values("$save_where/expr.txt", df_val[!, expr_col_name])
    write_real_values("$save_where/expr_minmax.txt", df_val[!, expr_minmax_col_name])

    save_where = joinpath(save_path, this_data_set_name, test_set)
    mkpath(save_where)
    write_fasta("$save_where/seqs.fa", df_test[!, seq_col_name])
    write_real_values("$save_where/expr.txt", df_test[!, expr_col_name])
    write_real_values("$save_where/expr_minmax.txt", df_test[!, expr_minmax_col_name])
end


seq_col_name="seq"
expr_col_name="expr"
source_name="gpro"
_shuffle_=true

function process_this_df(data_where, this_data_set_name; 
    seq_col_name="seq", expr_col_name="expr", 
    source_name="gpro", _shuffle_=true)

    df = CSV.read(data_where, DataFrame)
    if _shuffle_
        permuted_inds = shuffle(1:size(df, 1))
        df = df[permuted_inds, :]
    end

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

    # df_train[!,"expr_minmax"] = 
    #     (df_train_clipped .- df_train_min_expr) ./ max_minus_min
    # df_val[!,"expr_minmax"] = 
    #     (df_val_expr_clipped .- df_train_min_expr) ./ max_minus_min
    # df_test[!,"expr_minmax"] =
    #     (df_test_expr_clipped .- df_train_min_expr) ./ max_minus_min
    save_fasta_expr(df, df_train, df_val, df_test, 
        joinpath(source_name, this_data_set_name);
        seq_col_name=seq_col_name, expr_col_name=expr_col_name)
    return df_train, df_val, df_test
end


####################
this_data_set_name = "ecoli_50_wgan_diffusion_wanglab/"
data_where = joinpath(
    raw_data_path, gpro, "data",
    "$this_data_set_name/ecoli_natural50bp_expr.csv")
df_train, df_val, df_test = process_this_df(data_where, this_data_set_name)

####################

this_data_set_name = "ecoli_165_cgan_wanglab"
data_where = joinpath(
    raw_data_path, gpro, "data",
    "$this_data_set_name/ecoli_mpra_3_laco.csv")
df_train, df_val, df_test = process_this_df(data_where, this_data_set_name; seq_col_name="realB")


####################

this_data_set_name = "ecoli_165_cross_species_wanglab"
data_where = joinpath(
    raw_data_path, gpro, "data",
    "$this_data_set_name/ecoli_mpra_expr.csv")
df_train, df_val, df_test = process_this_df(data_where, this_data_set_name)

####################

this_data_set_name = "yeast_110_evolution_aviv"
data_where = joinpath(
    raw_data_path, gpro, "data",
    "$this_data_set_name/Random_test_tpu_model.csv")
df_train, df_val, df_test = process_this_df(data_where, this_data_set_name;
    seq_col_name="sequence", expr_col_name="Measured Expression");


#############################################################

using HDF5, Random, StatsBase
fid = h5open("processed_data_tanh2/gpro/ecoli_50_wgan_diffusion_wanglab/ecoli_wgan.hdf5", "w")
fid = h5open("processed_data_tanh2/gpro/ecoli_165_cgan_wanglab/ecoli_cgan.hdf5", "w")
fid = h5open("processed_data_tanh2/gpro/ecoli_165_cross_species_wanglab/ecoli_cross.hdf5", "w")
fid = h5open("processed_data_tanh2/gpro/yeast_110_evolution_aviv/yeast.hdf5", "w")

col_name = "realB" # 
col_name = "seq" # wgan, cross-species
col_name = "sequence" # yeast

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
