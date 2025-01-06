namespace MathUtils{
template<class T>
struct CSR{
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T>   values;
};

template <class T>
CSR<T> transformFullMatrix(const CSR<T>& smat)
{
    const auto& n       = smat.row_ptr.size() - 1;
    const auto& nnz     = smat.col_idx.size();
    const auto& values  = smat.values;
    const auto& row_ptr = smat.row_ptr;
    const auto& col_idx = smat.col_idx;

    std::vector<std::vector<T>>   values_tmp(n);
    std::vector<std::vector<int>> col_idx_tmp(n);

    std::vector<T>   m_values;
    std::vector<int> m_row_ptr(1, 0);
    std::vector<int> m_col_idx;

    int count = 0;
    for (int ii = 0; ii < n; ii++) {
        int start = row_ptr.at(ii);
        int end   = row_ptr.at(ii + 1);
        for (int j = start; j < end; j++) {
            int jj = col_idx.at(j);
            if (ii == jj) continue;

            values_tmp.at(jj).emplace_back(values.at(j));
            col_idx_tmp.at(jj).emplace_back(ii);
        }

        values_tmp.at(ii).insert(values_tmp.at(ii).end(), values.begin() + start, values.begin() + end);
        col_idx_tmp.at(ii).insert(col_idx_tmp.at(ii).end(), col_idx.begin() + start, col_idx.begin() + end);

        count += values_tmp.at(ii).size();
    }

    for (int ii = 0; ii < n; ii++) {
        m_values.insert(m_values.end(), values_tmp.at(ii).begin(), values_tmp.at(ii).end());
        m_col_idx.insert(m_col_idx.end(), col_idx_tmp.at(ii).begin(), col_idx_tmp.at(ii).end());
        m_row_ptr.emplace_back(m_values.size());
    }

    CSR<T> ret{m_row_ptr, m_col_idx, m_values};
    return ret;
}
} // namespace MathUtils
