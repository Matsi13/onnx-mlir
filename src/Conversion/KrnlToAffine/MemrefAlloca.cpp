/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlLoad.cpp - Lower KrnlLoadOp -----------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file convert memref.alloca to memref.alloc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
//#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

/// KrnlLoad will be lowered to std.load or affine.load, depending on whether
/// the access indices are all affine maps or not.
class MemrefAllocaLowering : public ConversionPattern {
public:
  explicit MemrefAllocaLowering(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, memref::AllocaOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto allocaOp = cast<memref::AllocaOp>(op);
    memref::AllocaOpAdaptor allocOpAdaptor(allocaOp);
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op,allocaOp.getType(),allocOpAdaptor.dynamicSizes(),allocaOp.alignmentAttr());


    return success();
  }
};

void populateLoweringMemrefAllocaOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<MemrefAllocaLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
