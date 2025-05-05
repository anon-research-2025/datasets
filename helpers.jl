
using HDF5, Random, StatsBase

"""
Take in a number (number of data points) and return a tuple of three vectors
    1st vector: training indices
    2nd vector: validation indices
    3rd vector: test indices
"""
function make_train_test_splits(num_reads; train_valid_test_ratio=[0.8,0.1,0.1])
    random_permuted_indices = shuffle(1:num_reads)
    train_end = floor(Int, num_reads*train_valid_test_ratio[1])
    valid_end = train_end + floor(Int, num_reads*train_valid_test_ratio[2])
    train_indices = random_permuted_indices[1:train_end]
    valid_indices = random_permuted_indices[train_end+1:valid_end]
    test_indices = random_permuted_indices[valid_end+1:end]
    return train_indices, valid_indices, test_indices
end


const quantile_max = 0.99
const quantile_min = 0.01

minmax_clip(val, min, max) = 
    val < min ? min : (val > max ? max : val)

get_q_max_expr(labels) = quantile.(eachrow(labels), quantile_max)
get_q_min_expr(labels) = quantile.(eachrow(labels), quantile_min)

function normalize_labelmat(labels, max_labels_each_row, min_labels_each_row)
    _diff_ = max_labels_each_row .- min_labels_each_row
    labels_normalized = (labels .- min_labels_each_row) ./ _diff_
    labels_normalized[labels_normalized .< 0] .= 0
    labels_normalized[labels_normalized .> 1] .= 1
    return labels_normalized
end

"""
labels: C x N matrix
    C: different values 
    N: number of data points
"""
function normalize_labels(labels, train_indices, valid_indices, test_indices)
    train_labels = labels[:,train_indices]
    valid_labels = labels[:,valid_indices]
    test_labels =  labels[:,test_indices]
    max_labels_each_row = get_q_max_expr(train_labels)
    min_labels_each_row = get_q_min_expr(train_labels)
    _normalize_(x) = 
        normalize_labelmat(x, max_labels_each_row, min_labels_each_row)
    train_labels = _normalize_(train_labels)
    valid_labels = _normalize_(valid_labels)
    test_labels =  _normalize_(test_labels)
    return train_labels, valid_labels, test_labels
end

const float_type = Float32

const _dummy_ = Dict('A'=>Array{float_type}([1, 0, 0, 0]), 
                'C'=>Array{float_type}([0, 1, 0, 0]),
                'G'=>Array{float_type}([0, 0, 1, 0]), 
                'T'=>Array{float_type}([0, 0, 0, 1]),
                'N'=>Array{float_type}([0, 0, 0, 0]));

function dna2dummy(dna_string::AbstractString; dummy::Dict=_dummy_, F=float_type)
    v = Array{F,2}(undef, (4, length(dna_string)));
    found_n = false
    @inbounds for (index, alphabet) in enumerate(dna_string)
        # start = (index-1)*4+1;
        # v[start:start+3] = dummy[uppercase(alphabet)];
        alphabet_here = uppercase(alphabet)
        alphabet_here == 'N' && (found_n = true)
        v[:,index] = dummy[alphabet_here];
    end
    # found_n && (@info "contains N in the string!")
    return v
end


#### trimming

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

# usage
#longest_common_prefix(df.seq)
#longest_common_suffix(df.seq)
#
#df.seq = trim_common_ends(df.seq)

function save_as_hdf5(
        df_train, df_val, df_test, save_name::String;
        folder_path="processed_data_tanh/deepTFBU", 
        df_seq_col_name="seq",
        )
    
    fid = h5open(joinpath(folder_path, "$save_name.hdf5"), "w")

    for (df_here, _name_) in zip([df_train, df_val, df_test], ["train", "valid", "test"])
        dna_array = Array{float_type, 3}(undef, (4, length(df_here[1,df_seq_col_name]), size(df_here,1)))
        for (_,i) in enumerate(axes(df_here, 1))
            dna_array[:, :, i] = dna2dummy(df_here[i,df_seq_col_name])
        end
    
        labels = Array{float_type, 2}(undef, (1, size(df_here,1)))
        for (_,i) in enumerate(axes(df_here, 1))
            labels[1, i] = float_type(df_here.expr_minmax[i])
        end

        fid["seq_1/$(_name_)"] = dna_array
        fid["expr/$(_name_)"] = labels
    end
    close(fid)
end