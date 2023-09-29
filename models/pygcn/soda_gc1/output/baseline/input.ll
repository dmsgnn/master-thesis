; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare i8* @malloc(i64)

declare void @free(i8*)

define void @forward_kernel(float* %0, float* %1, float* %2, float* %3, float* %4, float* %5) {
  %7 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %0, 0
  %8 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %7, float* %0, 1
  %9 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %8, i64 0, 2
  %10 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %9, i64 2708, 3, 0
  %11 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %10, i64 1433, 4, 0
  %12 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %11, i64 1433, 3, 1
  %13 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %12, i64 1, 4, 1
  %14 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %1, 0
  %15 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %14, float* %1, 1
  %16 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %15, i64 0, 2
  %17 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %16, i64 1433, 3, 0
  %18 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %17, i64 16, 4, 0
  %19 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %18, i64 16, 3, 1
  %20 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %19, i64 1, 4, 1
  %21 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %2, 0
  %22 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %21, float* %2, 1
  %23 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %22, i64 0, 2
  %24 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %23, i64 2708, 3, 0
  %25 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %24, i64 16, 4, 0
  %26 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %25, i64 16, 3, 1
  %27 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %26, i64 1, 4, 1
  %28 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %3, 0
  %29 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %28, float* %3, 1
  %30 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %29, i64 0, 2
  %31 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %30, i64 2708, 3, 0
  %32 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %31, i64 16, 4, 0
  %33 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %32, i64 16, 3, 1
  %34 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %33, i64 1, 4, 1
  %35 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %4, 0
  %36 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %35, float* %4, 1
  %37 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %36, i64 0, 2
  %38 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %37, i64 2708, 3, 0
  %39 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %38, i64 2708, 4, 0
  %40 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %39, i64 2708, 3, 1
  %41 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %40, i64 1, 4, 1
  %42 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } undef, float* %5, 0
  %43 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %42, float* %5, 1
  %44 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %43, i64 0, 2
  %45 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %44, i64 16, 3, 0
  %46 = insertvalue { float*, float*, i64, [1 x i64], [1 x i64] } %45, i64 1, 4, 0
  br label %47

47:                                               ; preds = %81, %6
  %48 = phi i64 [ %82, %81 ], [ 0, %6 ]
  %49 = icmp slt i64 %48, 2708
  br i1 %49, label %50, label %83

50:                                               ; preds = %79, %47
  %51 = phi i64 [ %80, %79 ], [ 0, %47 ]
  %52 = icmp slt i64 %51, 16
  br i1 %52, label %53, label %81

53:                                               ; preds = %56, %50
  %54 = phi i64 [ %78, %56 ], [ 0, %50 ]
  %55 = icmp slt i64 %54, 1433
  br i1 %55, label %56, label %79

56:                                               ; preds = %53
  %57 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %13, 1
  %58 = mul i64 %48, 1433
  %59 = add i64 %58, %54
  %60 = getelementptr float, float* %57, i64 %59
  %61 = load float, float* %60, align 4
  %62 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %20, 1
  %63 = mul i64 %54, 16
  %64 = add i64 %63, %51
  %65 = getelementptr float, float* %62, i64 %64
  %66 = load float, float* %65, align 4
  %67 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, 1
  %68 = mul i64 %48, 16
  %69 = add i64 %68, %51
  %70 = getelementptr float, float* %67, i64 %69
  %71 = load float, float* %70, align 4
  %72 = fmul float %61, %66
  %73 = fadd float %71, %72
  %74 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, 1
  %75 = mul i64 %48, 16
  %76 = add i64 %75, %51
  %77 = getelementptr float, float* %74, i64 %76
  store float %73, float* %77, align 4
  %78 = add i64 %54, 1
  br label %53

79:                                               ; preds = %53
  %80 = add i64 %51, 1
  br label %50

81:                                               ; preds = %50
  %82 = add i64 %48, 1
  br label %47

83:                                               ; preds = %47
  %84 = call i8* @malloc(i64 add (i64 ptrtoint (float* getelementptr (float, float* null, i32 43328) to i64), i64 64))
  %85 = bitcast i8* %84 to float*
  %86 = ptrtoint float* %85 to i64
  %87 = add i64 %86, 63
  %88 = urem i64 %87, 64
  %89 = sub i64 %87, %88
  %90 = inttoptr i64 %89 to float*
  %91 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } undef, float* %85, 0
  %92 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %91, float* %90, 1
  %93 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %92, i64 0, 2
  %94 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %93, i64 2708, 3, 0
  %95 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %94, i64 16, 3, 1
  %96 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %95, i64 16, 4, 0
  %97 = insertvalue { float*, float*, i64, [2 x i64], [2 x i64] } %96, i64 1, 4, 1
  %98 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %34, 3, 0
  %99 = mul i64 1, %98
  %100 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %34, 3, 1
  %101 = mul i64 %99, %100
  %102 = mul i64 %101, ptrtoint (float* getelementptr (float, float* null, i32 1) to i64)
  %103 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %34, 1
  %104 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %34, 2
  %105 = getelementptr float, float* %103, i64 %104
  %106 = getelementptr float, float* %90, i64 0
  call void @llvm.memcpy.p0f32.p0f32.i64(float* %106, float* %105, i64 %102, i1 false)
  br label %107

107:                                              ; preds = %139, %83
  %108 = phi i64 [ %140, %139 ], [ 0, %83 ]
  %109 = icmp slt i64 %108, 2708
  br i1 %109, label %110, label %141

110:                                              ; preds = %137, %107
  %111 = phi i64 [ %138, %137 ], [ 0, %107 ]
  %112 = icmp slt i64 %111, 16
  br i1 %112, label %113, label %139

113:                                              ; preds = %116, %110
  %114 = phi i64 [ %136, %116 ], [ 0, %110 ]
  %115 = icmp slt i64 %114, 2708
  br i1 %115, label %116, label %137

116:                                              ; preds = %113
  %117 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %41, 1
  %118 = mul i64 %108, 2708
  %119 = add i64 %118, %114
  %120 = getelementptr float, float* %117, i64 %119
  %121 = load float, float* %120, align 4
  %122 = extractvalue { float*, float*, i64, [2 x i64], [2 x i64] } %27, 1
  %123 = mul i64 %114, 16
  %124 = add i64 %123, %111
  %125 = getelementptr float, float* %122, i64 %124
  %126 = load float, float* %125, align 4
  %127 = mul i64 %108, 16
  %128 = add i64 %127, %111
  %129 = getelementptr float, float* %90, i64 %128
  %130 = load float, float* %129, align 4
  %131 = fmul float %121, %126
  %132 = fadd float %130, %131
  %133 = mul i64 %108, 16
  %134 = add i64 %133, %111
  %135 = getelementptr float, float* %90, i64 %134
  store float %132, float* %135, align 4
  %136 = add i64 %114, 1
  br label %113

137:                                              ; preds = %113
  %138 = add i64 %111, 1
  br label %110

139:                                              ; preds = %110
  %140 = add i64 %108, 1
  br label %107

141:                                              ; preds = %107
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0f32.p0f32.i64(float* noalias nocapture writeonly, float* noalias nocapture readonly, i64, i1 immarg) #0

attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
