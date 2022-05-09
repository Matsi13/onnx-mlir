#include <map>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"

#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/KrnlSupport.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"
#include "src/Conversion/KrnlToAffine/ConvertKrnlToAffine.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"



#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "krnl_to_affine"

using namespace mlir;
using namespace onnx_mlir;

namespace onnx_mlir {
namespace krnl {

/// KrnlLoad will be lowered to std.load or affine.load, depending on whether
/// the access indices are all affine maps or not.
class MemrefAllocaOpLowering : public ConversionPattern {
public:
  explicit MemrefAllocaOpLowering(TypeConverter &typeConverter, MLIRContext *context)
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
  patterns.insert<MemrefAllocaOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
