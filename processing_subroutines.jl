
# gpro = "gpro"
# train_set = "training_set"
# val_set = "validation_set"
# test_set = "test_set"

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


function process_this_df(df::DataFrame; expr_col_name="expr", _shuffle_=true)

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

    return df_train, df_val, df_test
end

