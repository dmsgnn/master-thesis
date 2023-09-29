module attributes {llvm.data_layout = "", soda.bambu.container_module, soda.container_module, torch.debug_module_name = "GCN"} {
  llvm.func @malloc(i64) -> !llvm.ptr<i8>
  llvm.func @forward_kernel(%arg0: !llvm.ptr<f32>, %arg1: !llvm.ptr<f32>, %arg2: !llvm.ptr<f32>, %arg3: !llvm.ptr<f32>, %arg4: !llvm.ptr<f32>, %arg5: !llvm.ptr<f32>) {
    %0 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %3 = llvm.mlir.constant(0 : index) : i64
    %4 = llvm.insertvalue %3, %2[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %5 = llvm.mlir.constant(2708 : index) : i64
    %6 = llvm.insertvalue %5, %4[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %7 = llvm.mlir.constant(1433 : index) : i64
    %8 = llvm.insertvalue %7, %6[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %9 = llvm.mlir.constant(1433 : index) : i64
    %10 = llvm.insertvalue %9, %8[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(1 : index) : i64
    %12 = llvm.insertvalue %11, %10[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %14 = llvm.insertvalue %arg1, %13[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %arg1, %14[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.mlir.constant(0 : index) : i64
    %17 = llvm.insertvalue %16, %15[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %18 = llvm.mlir.constant(1433 : index) : i64
    %19 = llvm.insertvalue %18, %17[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %20 = llvm.mlir.constant(16 : index) : i64
    %21 = llvm.insertvalue %20, %19[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %22 = llvm.mlir.constant(16 : index) : i64
    %23 = llvm.insertvalue %22, %21[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %24 = llvm.mlir.constant(1 : index) : i64
    %25 = llvm.insertvalue %24, %23[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %26 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %27 = llvm.insertvalue %arg2, %26[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.insertvalue %arg2, %27[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %29 = llvm.mlir.constant(0 : index) : i64
    %30 = llvm.insertvalue %29, %28[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.mlir.constant(2708 : index) : i64
    %32 = llvm.insertvalue %31, %30[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.mlir.constant(16 : index) : i64
    %34 = llvm.insertvalue %33, %32[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %35 = llvm.mlir.constant(16 : index) : i64
    %36 = llvm.insertvalue %35, %34[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %37 = llvm.mlir.constant(1 : index) : i64
    %38 = llvm.insertvalue %37, %36[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %39 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %40 = llvm.insertvalue %arg3, %39[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %41 = llvm.insertvalue %arg3, %40[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %42 = llvm.mlir.constant(0 : index) : i64
    %43 = llvm.insertvalue %42, %41[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.mlir.constant(2708 : index) : i64
    %45 = llvm.insertvalue %44, %43[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %46 = llvm.mlir.constant(16 : index) : i64
    %47 = llvm.insertvalue %46, %45[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.mlir.constant(16 : index) : i64
    %49 = llvm.insertvalue %48, %47[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.mlir.constant(1 : index) : i64
    %51 = llvm.insertvalue %50, %49[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %52 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %53 = llvm.insertvalue %arg4, %52[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %54 = llvm.insertvalue %arg4, %53[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %55 = llvm.mlir.constant(0 : index) : i64
    %56 = llvm.insertvalue %55, %54[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %57 = llvm.mlir.constant(2708 : index) : i64
    %58 = llvm.insertvalue %57, %56[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %59 = llvm.mlir.constant(2708 : index) : i64
    %60 = llvm.insertvalue %59, %58[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %61 = llvm.mlir.constant(2708 : index) : i64
    %62 = llvm.insertvalue %61, %60[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %63 = llvm.mlir.constant(1 : index) : i64
    %64 = llvm.insertvalue %63, %62[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %65 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)>
    %66 = llvm.insertvalue %arg5, %65[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %67 = llvm.insertvalue %arg5, %66[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %68 = llvm.mlir.constant(0 : index) : i64
    %69 = llvm.insertvalue %68, %67[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %70 = llvm.mlir.constant(16 : index) : i64
    %71 = llvm.insertvalue %70, %69[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %72 = llvm.mlir.constant(1 : index) : i64
    %73 = llvm.insertvalue %72, %71[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<1 x i64>, array<1 x i64>)> 
    %74 = llvm.mlir.constant(1433 : index) : i64
    %75 = llvm.mlir.constant(16 : index) : i64
    %76 = llvm.mlir.constant(1 : index) : i64
    %77 = llvm.mlir.constant(2708 : index) : i64
    %78 = llvm.mlir.constant(0 : index) : i64
    llvm.br ^bb1(%78 : i64)
  ^bb1(%79: i64):  // 2 preds: ^bb0, ^bb6
    %80 = llvm.icmp "slt" %79, %77 : i64
    llvm.cond_br %80, ^bb2(%78 : i64), ^bb7
  ^bb2(%81: i64):  // 2 preds: ^bb1, ^bb5
    %82 = llvm.icmp "slt" %81, %75 : i64
    llvm.cond_br %82, ^bb3(%78 : i64), ^bb6
  ^bb3(%83: i64):  // 2 preds: ^bb2, ^bb4
    %84 = llvm.icmp "slt" %83, %74 : i64
    llvm.cond_br %84, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %85 = llvm.extractvalue %12[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.mlir.constant(1433 : index) : i64
    %87 = llvm.mul %79, %86  : i64
    %88 = llvm.add %87, %83  : i64
    %89 = llvm.getelementptr %85[%88] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %90 = llvm.load %89 : !llvm.ptr<f32>
    %91 = llvm.extractvalue %25[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %92 = llvm.mlir.constant(16 : index) : i64
    %93 = llvm.mul %83, %92  : i64
    %94 = llvm.add %93, %81  : i64
    %95 = llvm.getelementptr %91[%94] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %96 = llvm.load %95 : !llvm.ptr<f32>
    %97 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %98 = llvm.mlir.constant(16 : index) : i64
    %99 = llvm.mul %79, %98  : i64
    %100 = llvm.add %99, %81  : i64
    %101 = llvm.getelementptr %97[%100] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %102 = llvm.load %101 : !llvm.ptr<f32>
    %103 = llvm.fmul %90, %96  : f32
    %104 = llvm.fadd %102, %103  : f32
    %105 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mlir.constant(16 : index) : i64
    %107 = llvm.mul %79, %106  : i64
    %108 = llvm.add %107, %81  : i64
    %109 = llvm.getelementptr %105[%108] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %104, %109 : !llvm.ptr<f32>
    %110 = llvm.add %83, %76  : i64
    llvm.br ^bb3(%110 : i64)
  ^bb5:  // pred: ^bb3
    %111 = llvm.add %81, %76  : i64
    llvm.br ^bb2(%111 : i64)
  ^bb6:  // pred: ^bb2
    %112 = llvm.add %79, %76  : i64
    llvm.br ^bb1(%112 : i64)
  ^bb7:  // pred: ^bb1
    %113 = llvm.mlir.constant(2708 : index) : i64
    %114 = llvm.mlir.constant(16 : index) : i64
    %115 = llvm.mlir.constant(1 : index) : i64
    %116 = llvm.mlir.constant(43328 : index) : i64
    %117 = llvm.mlir.null : !llvm.ptr<f32>
    %118 = llvm.getelementptr %117[43328] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %119 = llvm.ptrtoint %118 : !llvm.ptr<f32> to i64
    %120 = llvm.mlir.constant(64 : index) : i64
    %121 = llvm.add %119, %120  : i64
    %122 = llvm.call @malloc(%121) : (i64) -> !llvm.ptr<i8>
    %123 = llvm.bitcast %122 : !llvm.ptr<i8> to !llvm.ptr<f32>
    %124 = llvm.ptrtoint %123 : !llvm.ptr<f32> to i64
    %125 = llvm.mlir.constant(1 : index) : i64
    %126 = llvm.sub %120, %125  : i64
    %127 = llvm.add %124, %126  : i64
    %128 = llvm.urem %127, %120  : i64
    %129 = llvm.sub %127, %128  : i64
    %130 = llvm.inttoptr %129 : i64 to !llvm.ptr<f32>
    %131 = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
    %132 = llvm.insertvalue %123, %131[0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %133 = llvm.insertvalue %130, %132[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %134 = llvm.mlir.constant(0 : index) : i64
    %135 = llvm.insertvalue %134, %133[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.insertvalue %113, %135[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %137 = llvm.insertvalue %114, %136[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %138 = llvm.insertvalue %114, %137[4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %139 = llvm.insertvalue %115, %138[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %140 = llvm.mlir.constant(1 : index) : i64
    %141 = llvm.extractvalue %51[3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %142 = llvm.mul %140, %141  : i64
    %143 = llvm.extractvalue %51[3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %144 = llvm.mul %142, %143  : i64
    %145 = llvm.mlir.null : !llvm.ptr<f32>
    %146 = llvm.getelementptr %145[1] : (!llvm.ptr<f32>) -> !llvm.ptr<f32>
    %147 = llvm.ptrtoint %146 : !llvm.ptr<f32> to i64
    %148 = llvm.mul %144, %147  : i64
    %149 = llvm.extractvalue %51[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %150 = llvm.extractvalue %51[2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.getelementptr %149[%150] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %152 = llvm.getelementptr %130[%134] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %153 = llvm.mlir.constant(false) : i1
    "llvm.intr.memcpy"(%152, %151, %148, %153) : (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i1) -> ()
    llvm.br ^bb8(%78 : i64)
  ^bb8(%154: i64):  // 2 preds: ^bb7, ^bb13
    %155 = llvm.icmp "slt" %154, %77 : i64
    llvm.cond_br %155, ^bb9(%78 : i64), ^bb14
  ^bb9(%156: i64):  // 2 preds: ^bb8, ^bb12
    %157 = llvm.icmp "slt" %156, %75 : i64
    llvm.cond_br %157, ^bb10(%78 : i64), ^bb13
  ^bb10(%158: i64):  // 2 preds: ^bb9, ^bb11
    %159 = llvm.icmp "slt" %158, %77 : i64
    llvm.cond_br %159, ^bb11, ^bb12
  ^bb11:  // pred: ^bb10
    %160 = llvm.extractvalue %64[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.mlir.constant(2708 : index) : i64
    %162 = llvm.mul %154, %161  : i64
    %163 = llvm.add %162, %158  : i64
    %164 = llvm.getelementptr %160[%163] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %165 = llvm.load %164 : !llvm.ptr<f32>
    %166 = llvm.extractvalue %38[1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
    %167 = llvm.mlir.constant(16 : index) : i64
    %168 = llvm.mul %158, %167  : i64
    %169 = llvm.add %168, %156  : i64
    %170 = llvm.getelementptr %166[%169] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %171 = llvm.load %170 : !llvm.ptr<f32>
    %172 = llvm.mlir.constant(16 : index) : i64
    %173 = llvm.mul %154, %172  : i64
    %174 = llvm.add %173, %156  : i64
    %175 = llvm.getelementptr %130[%174] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    %176 = llvm.load %175 : !llvm.ptr<f32>
    %177 = llvm.fmul %165, %171  : f32
    %178 = llvm.fadd %176, %177  : f32
    %179 = llvm.mlir.constant(16 : index) : i64
    %180 = llvm.mul %154, %179  : i64
    %181 = llvm.add %180, %156  : i64
    %182 = llvm.getelementptr %130[%181] : (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
    llvm.store %178, %182 : !llvm.ptr<f32>
    %183 = llvm.add %158, %76  : i64
    llvm.br ^bb10(%183 : i64)
  ^bb12:  // pred: ^bb10
    %184 = llvm.add %156, %76  : i64
    llvm.br ^bb9(%184 : i64)
  ^bb13:  // pred: ^bb9
    %185 = llvm.add %154, %76  : i64
    llvm.br ^bb8(%185 : i64)
  ^bb14:  // pred: ^bb8
    llvm.return
  }
}

