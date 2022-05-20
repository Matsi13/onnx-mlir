/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- KrnlLoad.cpp - Lower KrnlLoadOp -----------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlGlobalOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

/// KrnlLoad will be lowered to std.load or affine.load, depending on whether
/// the access indices are all affine maps or not.
class KrnlGlobalLowering : public ConversionPattern {
public:
  explicit KrnlGlobalLowering(TypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlGlobalOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto globalOp = cast<KrnlGlobalOp>(op);
    // KrnlGlobalOpAdaptor operandAdaptor(globalOp);
    func::FuncOp funcOp = op-> getParentOfType<func::FuncOp>();
    OpBuilder builder(funcOp);
    memref::GlobalOp memGlobal = builder.create<memref::GlobalOp>(funcOp.getLoc(),globalOp.nameAttr().getValue(),
                                  StringAttr(),globalOp.output().getType().dyn_cast<MemRefType>(),globalOp.valueAttr(),
                                  false, IntegerAttr());
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(op,globalOp.output().getType(),memGlobal.sym_nameAttr().getValue());


    

    return success();
  }
};

void populateLoweringKrnlGlobalOpPattern(TypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlGlobalLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
