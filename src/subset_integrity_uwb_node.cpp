// Subset-consensus integrity variant.
//
// This translation unit reuses the production WLS+LC implementation and enables
// the RANSAC-style anchor subset detector at compile time. Keeping it as a
// separate executable lets us compare both integrity algorithms on the same bag.

#define UWB_SUBSET_CONSENSUS_INTEGRITY 1
#include "integrity_uwb_node.cpp"
