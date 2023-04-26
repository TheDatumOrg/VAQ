#ifndef FPTREE_HPP
#define FPTREE_HPP

#include "../Types.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <utility>

using Item = uint;
using Transaction = std::vector<Item>;
using TransformedPrefixPath = std::pair<std::vector<Item>, uint64_t>;
using Pattern = std::pair<std::set<Item>, uint64_t>;
using SetPattern = std::set<Pattern>;

struct FPNode {
    const Item item;
    uint64_t frequency;
    std::shared_ptr<FPNode> node_link;
    std::weak_ptr<FPNode> parent;
    std::vector<std::shared_ptr<FPNode>> children;

    FPNode(const Item&, const std::shared_ptr<FPNode>&);
};

struct FPTree {
    std::shared_ptr<FPNode> root;
    std::map<Item, std::shared_ptr<FPNode>> header_table;
    uint64_t minimum_support_threshold;

    FPTree(const std::vector<Transaction>& transactions, uint64_t minimum_support_threshold);
    FPTree(CodebookType transactions, uint64_t minimum_support_threshold, const std::vector<int> &centroidsNum);

    bool empty() const;
};


SetPattern fptree_growth(const FPTree&);


#endif  // FPTREE_HPP
