/*!
 * \file tl/metal/op/scan.cc
 * \brief Metal shared-memory inclusive scan lowering.
 */

#include "op/scan.h"

#include <tvm/tirx/builtin.h>

#include "metal/op/builtin.h"
#include "metal/target_utils.h"
#include "op/utils.h"

namespace tvm {
namespace tl {
namespace {

Stmt LowerMetalCumSum(const CumSumOpNode &op, const LowerArgs &args,
                      arith::Analyzer *analyzer) {
  CHECK(IsSharedBuffer(op.src) && IsSharedBuffer(op.dst), ValueError)
      << "Metal cumsum requires shared or shared.dyn source and destination";
  CHECK(op.src->shape.size() == 1 && op.dst->shape.size() == 1 && op.dim == 0,
        ValueError)
      << "Metal cumsum currently supports only one-dimensional buffers";
  CHECK(op.src->dtype == op.dst->dtype &&
            (op.src->dtype == DataType::Int(32) ||
             op.src->dtype == DataType::Float(32)),
        ValueError)
      << "Metal cumsum requires matching int32 or float32 buffers";
  CHECK(op.src->strides.empty() && op.dst->strides.empty() &&
            analyzer->CanProveEqual(op.src->elem_offset, 0) &&
            analyzer->CanProveEqual(op.dst->elem_offset, 0),
        ValueError)
      << "Metal cumsum requires compact buffers with zero elem_offset";

  Range src = op.srcRegion_->region[0];
  Range dst = op.dstRegion_->region[0];
  const auto *length = src->extent.as<IntImmNode>();
  CHECK(length && length->value > 0 &&
            analyzer->CanProveEqual(src->extent, dst->extent),
        ValueError)
      << "Metal cumsum requires matching positive constant region lengths";
  CHECK(analyzer->CanProve(src->min >= 0) &&
            analyzer->CanProve(src->min + src->extent <= op.src->shape[0]) &&
            analyzer->CanProve(dst->min >= 0) &&
            analyzer->CanProve(dst->min + dst->extent <= op.dst->shape[0]),
        ValueError)
      << "Metal cumsum requires regions provably inside their buffers";
  if (op.src->data.same_as(op.dst->data)) {
    CHECK((analyzer->CanProveEqual(src->min, dst->min) ||
           analyzer->CanProve(src->min + src->extent <= dst->min) ||
           analyzer->CanProve(dst->min + dst->extent <= src->min)),
          ValueError)
        << "Metal cumsum requires identical or disjoint regions when source "
           "and destination alias";
  }

  int width = TargetMetalGetWarpSize(args.target);
  CHECK(analyzer->CanProveEqual(args.thread_bounds->min, 0) &&
            analyzer->CanProve(args.thread_bounds->extent >= width) &&
            analyzer->CanProveEqual(floormod(args.thread_bounds->extent, width),
                                    0),
        ValueError)
      << "Metal cumsum requires a zero-based thread extent divisible by "
      << width;

  // Keep buffer effects visible through access pointers, but encapsulate the
  // lane-level scan so loop unswitching and ThreadSync cannot introduce a
  // threadgroup barrier inside the first-SIMD-group branch.
  Stmt scan =
      Evaluate(Call(DataType::Handle(), metal_cumsum(),
                    {MakeAccessPtrFromRegion(op.srcRegion_, kAccessRead),
                     MakeAccessPtrFromRegion(op.dstRegion_, kAccessWrite),
                     src->extent, args.thread_index, Bool(op.reverse)}));
  auto sync = []() {
    return Evaluate(Call(DataType::Int(32), tirx::builtin::tvm_storage_sync(),
                         {StringImm("shared")}));
  };
  return SeqStmt({sync(), scan, sync()});
}

bool RegisterMetalScan() {
  RegisterCumSumImpl(
      CumSumImpl{"metal.CumSum", TargetIsMetal, LowerMetalCumSum});
  return true;
}

const bool metal_scan_registered = RegisterMetalScan();

} // namespace
} // namespace tl
} // namespace tvm
