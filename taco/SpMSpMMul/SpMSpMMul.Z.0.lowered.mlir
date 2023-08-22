module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @SpMSpMMul.Z.0.main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64, %arg3: i64, %arg4: i64, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: !llvm.ptr, %arg11: !llvm.ptr, %arg12: i64, %arg13: i64, %arg14: i64, %arg15: !llvm.struct<(array<2 x i64>, array<3 x i64>)>, %arg16: !llvm.ptr, %arg17: !llvm.ptr, %arg18: i64, %arg19: i64, %arg20: i64, %arg21: !llvm.ptr, %arg22: !llvm.ptr, %arg23: i64, %arg24: i64, %arg25: i64, %arg26: !llvm.ptr, %arg27: !llvm.ptr, %arg28: i64, %arg29: i64, %arg30: i64, %arg31: !llvm.struct<(array<2 x i64>, array<3 x i64>)>) -> !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(array<2 x i64>, array<1 x i64>)>)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.insertvalue %arg3, %3[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.insertvalue %arg4, %4[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.insertvalue %arg5, %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.insertvalue %arg6, %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.insertvalue %arg7, %8[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.insertvalue %arg8, %9[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.insertvalue %arg9, %10[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.insertvalue %arg10, %12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.insertvalue %arg11, %13[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.insertvalue %arg12, %14[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.insertvalue %arg13, %15[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.insertvalue %arg14, %16[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.insertvalue %arg16, %18[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.insertvalue %arg17, %19[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.insertvalue %arg18, %20[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.insertvalue %arg19, %21[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.insertvalue %arg20, %22[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.insertvalue %arg21, %24[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.insertvalue %arg22, %25[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.insertvalue %arg23, %26[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.insertvalue %arg24, %27[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.insertvalue %arg25, %28[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.insertvalue %arg26, %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.insertvalue %arg27, %31[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.insertvalue %arg28, %32[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.insertvalue %arg29, %33[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.insertvalue %arg30, %34[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.mlir.constant(0 : index) : i64
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.mlir.constant(240 : index) : i64
    %39 = llvm.mlir.constant(16 : index) : i64
    %40 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %41 = llvm.mlir.constant(15 : index) : i64
    %42 = llvm.mlir.constant(0 : i64) : i64
    %43 = llvm.mlir.constant(15 : i64) : i64
    %44 = llvm.mlir.constant(16 : i64) : i64
    %45 = llvm.mlir.constant(240 : i64) : i64
    %46 = llvm.mlir.constant(1 : index) : i64
    %47 = llvm.mlir.constant(240 : index) : i64
    %48 = llvm.mlir.constant(0 : index) : i64
    %49 = llvm.mlir.null : !llvm.ptr
    %50 = llvm.getelementptr %49[240] : (!llvm.ptr) -> !llvm.ptr, f32
    %51 = llvm.ptrtoint %50 : !llvm.ptr to i64
    %52 = llvm.call @malloc(%51) : (i64) -> !llvm.ptr
    %53 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %54 = llvm.insertvalue %52, %53[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %55 = llvm.insertvalue %52, %54[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %56 = llvm.insertvalue %36, %55[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %57 = llvm.insertvalue %38, %56[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %58 = llvm.insertvalue %37, %57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %59 = llvm.mlir.undef : !llvm.struct<(array<2 x i64>, array<1 x i64>)>
    %60 = llvm.insertvalue %42, %59[1, 0] : !llvm.struct<(array<2 x i64>, array<1 x i64>)> 
    %61 = llvm.insertvalue %43, %60[0, 0] : !llvm.struct<(array<2 x i64>, array<1 x i64>)> 
    %62 = llvm.insertvalue %44, %61[0, 1] : !llvm.struct<(array<2 x i64>, array<1 x i64>)> 
    llvm.br ^bb1(%48 : i64)
  ^bb1(%63: i64):  // 2 preds: ^bb0, ^bb2
    %64 = llvm.icmp "slt" %63, %47 : i64
    llvm.cond_br %64, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %65 = llvm.getelementptr %52[%63] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %40, %65 : f32, !llvm.ptr
    %66 = llvm.add %63, %46  : i64
    llvm.br ^bb1(%66 : i64)
  ^bb3:  // pred: ^bb1
    %67 = llvm.insertvalue %45, %62[1, 0] : !llvm.struct<(array<2 x i64>, array<1 x i64>)> 
    llvm.br ^bb4(%48 : i64)
  ^bb4(%68: i64):  // 2 preds: ^bb3, ^bb15
    %69 = llvm.icmp "slt" %68, %41 : i64
    llvm.cond_br %69, ^bb5, ^bb16
  ^bb5:  // pred: ^bb4
    llvm.br ^bb6(%48 : i64)
  ^bb6(%70: i64):  // 2 preds: ^bb5, ^bb14
    %71 = llvm.icmp "slt" %70, %39 : i64
    llvm.cond_br %71, ^bb7, ^bb15
  ^bb7:  // pred: ^bb6
    %72 = llvm.mul %68, %39  : i64
    %73 = llvm.add %72, %70  : i64
    %74 = llvm.getelementptr %52[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %75 = llvm.load %74 : !llvm.ptr -> f32
    %76 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %77 = llvm.getelementptr %76[%68] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %78 = llvm.load %77 : !llvm.ptr -> i64
    %79 = llvm.add %68, %46  : i64
    %80 = llvm.extractvalue %5[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %81 = llvm.getelementptr %80[%79] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %82 = llvm.load %81 : !llvm.ptr -> i64
    %83 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %84 = llvm.getelementptr %83[%70] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %85 = llvm.load %84 : !llvm.ptr -> i64
    %86 = llvm.add %70, %46  : i64
    %87 = llvm.extractvalue %23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %88 = llvm.getelementptr %87[%86] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %89 = llvm.load %88 : !llvm.ptr -> i64
    llvm.br ^bb8(%78, %85, %75 : i64, i64, f32)
  ^bb8(%90: i64, %91: i64, %92: f32):  // 2 preds: ^bb7, ^bb13
    %93 = llvm.icmp "ult" %90, %82 : i64
    %94 = llvm.icmp "ult" %91, %89 : i64
    %95 = llvm.and %93, %94  : i1
    llvm.cond_br %95, ^bb9(%90, %91, %92 : i64, i64, f32), ^bb14
  ^bb9(%96: i64, %97: i64, %98: f32):  // pred: ^bb8
    %99 = llvm.extractvalue %11[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %100 = llvm.getelementptr %99[%96] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %101 = llvm.load %100 : !llvm.ptr -> i64
    %102 = llvm.extractvalue %29[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %103 = llvm.getelementptr %102[%97] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %104 = llvm.load %103 : !llvm.ptr -> i64
    %105 = llvm.icmp "ult" %104, %101 : i64
    %106 = llvm.select %105, %104, %101 : i1, i64
    %107 = llvm.icmp "eq" %101, %106 : i64
    %108 = llvm.icmp "eq" %104, %106 : i64
    %109 = llvm.and %107, %108  : i1
    llvm.cond_br %109, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %110 = llvm.extractvalue %17[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %111 = llvm.getelementptr %110[%96] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %112 = llvm.load %111 : !llvm.ptr -> f32
    %113 = llvm.extractvalue %35[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %114 = llvm.getelementptr %113[%97] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %115 = llvm.load %114 : !llvm.ptr -> f32
    %116 = llvm.fmul %112, %115  : f32
    %117 = llvm.fadd %98, %116  : f32
    llvm.br ^bb12(%117 : f32)
  ^bb11:  // pred: ^bb9
    llvm.br ^bb12(%98 : f32)
  ^bb12(%118: f32):  // 2 preds: ^bb10, ^bb11
    llvm.br ^bb13
  ^bb13:  // pred: ^bb12
    %119 = llvm.icmp "eq" %101, %106 : i64
    %120 = llvm.add %96, %46  : i64
    %121 = llvm.select %119, %120, %96 : i1, i64
    %122 = llvm.icmp "eq" %104, %106 : i64
    %123 = llvm.add %97, %46  : i64
    %124 = llvm.select %122, %123, %97 : i1, i64
    llvm.br ^bb8(%121, %124, %118 : i64, i64, f32)
  ^bb14:  // pred: ^bb8
    %125 = llvm.getelementptr %52[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %92, %125 : f32, !llvm.ptr
    %126 = llvm.add %70, %46  : i64
    llvm.br ^bb6(%126 : i64)
  ^bb15:  // pred: ^bb6
    %127 = llvm.add %68, %46  : i64
    llvm.br ^bb4(%127 : i64)
  ^bb16:  // pred: ^bb4
    %128 = llvm.mlir.undef : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(array<2 x i64>, array<1 x i64>)>)>
    %129 = llvm.insertvalue %58, %128[0] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(array<2 x i64>, array<1 x i64>)>)> 
    %130 = llvm.insertvalue %67, %129[1] : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(array<2 x i64>, array<1 x i64>)>)> 
    llvm.return %130 : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(array<2 x i64>, array<1 x i64>)>)>
  }
  llvm.func @_mlir_ciface_SpMSpMMul.Z.0.main(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: !llvm.ptr, %arg4: !llvm.struct<(array<2 x i64>, array<3 x i64>)>, %arg5: !llvm.ptr, %arg6: !llvm.ptr, %arg7: !llvm.ptr, %arg8: !llvm.struct<(array<2 x i64>, array<3 x i64>)>) attributes {llvm.emit_c_interface} {
    %0 = llvm.load %arg1 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %1 = llvm.extractvalue %0[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %2 = llvm.extractvalue %0[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %3 = llvm.extractvalue %0[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %4 = llvm.extractvalue %0[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %5 = llvm.extractvalue %0[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %6 = llvm.load %arg2 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %8 = llvm.extractvalue %6[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %9 = llvm.extractvalue %6[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %10 = llvm.extractvalue %6[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %11 = llvm.extractvalue %6[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %12 = llvm.load %arg3 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %13 = llvm.extractvalue %12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %14 = llvm.extractvalue %12[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %15 = llvm.extractvalue %12[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %16 = llvm.extractvalue %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %17 = llvm.extractvalue %12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %18 = llvm.load %arg5 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %19 = llvm.extractvalue %18[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %20 = llvm.extractvalue %18[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %21 = llvm.extractvalue %18[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %22 = llvm.extractvalue %18[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %23 = llvm.extractvalue %18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %24 = llvm.load %arg6 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %26 = llvm.extractvalue %24[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %27 = llvm.extractvalue %24[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %28 = llvm.extractvalue %24[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %29 = llvm.extractvalue %24[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %30 = llvm.load %arg7 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %31 = llvm.extractvalue %30[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %32 = llvm.extractvalue %30[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %33 = llvm.extractvalue %30[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %34 = llvm.extractvalue %30[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %35 = llvm.extractvalue %30[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    %36 = llvm.call @SpMSpMMul.Z.0.main(%1, %2, %3, %4, %5, %7, %8, %9, %10, %11, %13, %14, %15, %16, %17, %arg4, %19, %20, %21, %22, %23, %25, %26, %27, %28, %29, %31, %32, %33, %34, %35, %arg8) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.struct<(array<2 x i64>, array<3 x i64>)>, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, !llvm.struct<(array<2 x i64>, array<3 x i64>)>) -> !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(array<2 x i64>, array<1 x i64>)>)>
    llvm.store %36, %arg0 : !llvm.struct<(struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, struct<(array<2 x i64>, array<1 x i64>)>)>, !llvm.ptr
    llvm.return
  }
}

