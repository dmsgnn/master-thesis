; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

declare void @free(ptr)

define { { ptr, ptr, i64, [1 x i64], [1 x i64] }, { [2 x i64], [1 x i64] } } @SpMSpMMul.Z.0.main(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, ptr %10, ptr %11, i64 %12, i64 %13, i64 %14, { [2 x i64], [3 x i64] } %15, ptr %16, ptr %17, i64 %18, i64 %19, i64 %20, ptr %21, ptr %22, i64 %23, i64 %24, i64 %25, ptr %26, ptr %27, i64 %28, i64 %29, i64 %30, { [2 x i64], [3 x i64] } %31) {
  %33 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %0, 0
  %34 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %33, ptr %1, 1
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, i64 %2, 2
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, i64 %3, 3, 0
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, i64 %4, 4, 0
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %5, 0
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, ptr %6, 1
  %40 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, i64 %7, 2
  %41 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, i64 %8, 3, 0
  %42 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %41, i64 %9, 4, 0
  %43 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %10, 0
  %44 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %43, ptr %11, 1
  %45 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %44, i64 %12, 2
  %46 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %45, i64 %13, 3, 0
  %47 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %46, i64 %14, 4, 0
  %48 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %16, 0
  %49 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %48, ptr %17, 1
  %50 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %49, i64 %18, 2
  %51 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %50, i64 %19, 3, 0
  %52 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %51, i64 %20, 4, 0
  %53 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %21, 0
  %54 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %53, ptr %22, 1
  %55 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %54, i64 %23, 2
  %56 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %55, i64 %24, 3, 0
  %57 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %56, i64 %25, 4, 0
  %58 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %26, 0
  %59 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %58, ptr %27, 1
  %60 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %59, i64 %28, 2
  %61 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %60, i64 %29, 3, 0
  %62 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %61, i64 %30, 4, 0
  %63 = call ptr @malloc(i64 ptrtoint (ptr getelementptr (float, ptr null, i32 240) to i64))
  %64 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } undef, ptr %63, 0
  %65 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %64, ptr %63, 1
  %66 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %65, i64 0, 2
  %67 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %66, i64 240, 3, 0
  %68 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %67, i64 1, 4, 0
  br label %69

69:                                               ; preds = %72, %32
  %70 = phi i64 [ %74, %72 ], [ 0, %32 ]
  %71 = icmp slt i64 %70, 240
  br i1 %71, label %72, label %75

72:                                               ; preds = %69
  %73 = getelementptr float, ptr %63, i64 %70
  store float 0.000000e+00, ptr %73, align 4
  %74 = add i64 %70, 1
  br label %69

75:                                               ; preds = %69
  br label %76

76:                                               ; preds = %146, %75
  %77 = phi i64 [ %147, %146 ], [ 0, %75 ]
  %78 = icmp slt i64 %77, 15
  br i1 %78, label %79, label %148

79:                                               ; preds = %76
  br label %80

80:                                               ; preds = %143, %79
  %81 = phi i64 [ %145, %143 ], [ 0, %79 ]
  %82 = icmp slt i64 %81, 16
  br i1 %82, label %83, label %146

83:                                               ; preds = %80
  %84 = mul i64 %77, 16
  %85 = add i64 %84, %81
  %86 = getelementptr float, ptr %63, i64 %85
  %87 = load float, ptr %86, align 4
  %88 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %89 = getelementptr i64, ptr %88, i64 %77
  %90 = load i64, ptr %89, align 4
  %91 = add i64 %77, 1
  %92 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, 1
  %93 = getelementptr i64, ptr %92, i64 %91
  %94 = load i64, ptr %93, align 4
  %95 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 1
  %96 = getelementptr i64, ptr %95, i64 %81
  %97 = load i64, ptr %96, align 4
  %98 = add i64 %81, 1
  %99 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %52, 1
  %100 = getelementptr i64, ptr %99, i64 %98
  %101 = load i64, ptr %100, align 4
  br label %102

102:                                              ; preds = %136, %83
  %103 = phi i64 [ %139, %136 ], [ %90, %83 ]
  %104 = phi i64 [ %142, %136 ], [ %97, %83 ]
  %105 = phi float [ %135, %136 ], [ %87, %83 ]
  %106 = icmp ult i64 %103, %94
  %107 = icmp ult i64 %104, %101
  %108 = and i1 %106, %107
  br i1 %108, label %109, label %143

109:                                              ; preds = %102
  %110 = phi i64 [ %103, %102 ]
  %111 = phi i64 [ %104, %102 ]
  %112 = phi float [ %105, %102 ]
  %113 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %42, 1
  %114 = getelementptr i64, ptr %113, i64 %110
  %115 = load i64, ptr %114, align 4
  %116 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %57, 1
  %117 = getelementptr i64, ptr %116, i64 %111
  %118 = load i64, ptr %117, align 4
  %119 = icmp ult i64 %118, %115
  %120 = select i1 %119, i64 %118, i64 %115
  %121 = icmp eq i64 %115, %120
  %122 = icmp eq i64 %118, %120
  %123 = and i1 %121, %122
  br i1 %123, label %124, label %133

124:                                              ; preds = %109
  %125 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %47, 1
  %126 = getelementptr float, ptr %125, i64 %110
  %127 = load float, ptr %126, align 4
  %128 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %62, 1
  %129 = getelementptr float, ptr %128, i64 %111
  %130 = load float, ptr %129, align 4
  %131 = fmul float %127, %130
  %132 = fadd float %112, %131
  br label %134

133:                                              ; preds = %109
  br label %134

134:                                              ; preds = %124, %133
  %135 = phi float [ %112, %133 ], [ %132, %124 ]
  br label %136

136:                                              ; preds = %134
  %137 = icmp eq i64 %115, %120
  %138 = add i64 %110, 1
  %139 = select i1 %137, i64 %138, i64 %110
  %140 = icmp eq i64 %118, %120
  %141 = add i64 %111, 1
  %142 = select i1 %140, i64 %141, i64 %111
  br label %102

143:                                              ; preds = %102
  %144 = getelementptr float, ptr %63, i64 %85
  store float %105, ptr %144, align 4
  %145 = add i64 %81, 1
  br label %80

146:                                              ; preds = %80
  %147 = add i64 %77, 1
  br label %76

148:                                              ; preds = %76
  %149 = insertvalue { { ptr, ptr, i64, [1 x i64], [1 x i64] }, { [2 x i64], [1 x i64] } } undef, { ptr, ptr, i64, [1 x i64], [1 x i64] } %68, 0
  %150 = insertvalue { { ptr, ptr, i64, [1 x i64], [1 x i64] }, { [2 x i64], [1 x i64] } } %149, { [2 x i64], [1 x i64] } { [2 x i64] [i64 15, i64 16], [1 x i64] [i64 240] }, 1
  ret { { ptr, ptr, i64, [1 x i64], [1 x i64] }, { [2 x i64], [1 x i64] } } %150
}

define void @_mlir_ciface_SpMSpMMul.Z.0.main(ptr %0, ptr %1, ptr %2, ptr %3, { [2 x i64], [3 x i64] } %4, ptr %5, ptr %6, ptr %7, { [2 x i64], [3 x i64] } %8) {
  %10 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %11 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 1
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 2
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 3, 0
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, 4, 0
  %16 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %17 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 0
  %18 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 1
  %19 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 2
  %20 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 3, 0
  %21 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 4, 0
  %22 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %3, align 8
  %23 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 0
  %24 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 1
  %25 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 2
  %26 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 3, 0
  %27 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, 4, 0
  %28 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %5, align 8
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 0
  %30 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 1
  %31 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 2
  %32 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 3, 0
  %33 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, 4, 0
  %34 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %6, align 8
  %35 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 0
  %36 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 1
  %37 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 2
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 3, 0
  %39 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %34, 4, 0
  %40 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %7, align 8
  %41 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 0
  %42 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 1
  %43 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 2
  %44 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 3, 0
  %45 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %40, 4, 0
  %46 = call { { ptr, ptr, i64, [1 x i64], [1 x i64] }, { [2 x i64], [1 x i64] } } @SpMSpMMul.Z.0.main(ptr %11, ptr %12, i64 %13, i64 %14, i64 %15, ptr %17, ptr %18, i64 %19, i64 %20, i64 %21, ptr %23, ptr %24, i64 %25, i64 %26, i64 %27, { [2 x i64], [3 x i64] } %4, ptr %29, ptr %30, i64 %31, i64 %32, i64 %33, ptr %35, ptr %36, i64 %37, i64 %38, i64 %39, ptr %41, ptr %42, i64 %43, i64 %44, i64 %45, { [2 x i64], [3 x i64] } %8)
  store { { ptr, ptr, i64, [1 x i64], [1 x i64] }, { [2 x i64], [1 x i64] } } %46, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
