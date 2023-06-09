; ModuleID = 'input.ll'
source_filename = "LLVMDialectModule"

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef i8* @malloc(i64 noundef) local_unnamed_addr #0

; Function Attrs: nofree nounwind
define void @forward_kernel(float* nocapture readonly %0, float* nocapture readonly %1, float* nocapture %2, float* nocapture readonly %3, float* nocapture readonly %4, float* nocapture readnone %5) local_unnamed_addr #1 {
  br label %.preheader4

.preheader4:                                      ; preds = %6, %30
  %7 = phi i64 [ 0, %6 ], [ %31, %30 ]
  %8 = mul nuw nsw i64 %7, 1433
  %9 = shl nuw nsw i64 %7, 4
  br label %.preheader3

.preheader3:                                      ; preds = %.preheader4, %27
  %10 = phi i64 [ 0, %.preheader4 ], [ %28, %27 ]
  %11 = add nuw nsw i64 %10, %9
  %12 = getelementptr float, float* %2, i64 %11
  %.pre = load float, float* %12, align 4
  br label %13

13:                                               ; preds = %.preheader3, %13
  %14 = phi float [ %.pre, %.preheader3 ], [ %24, %13 ]
  %15 = phi i64 [ 0, %.preheader3 ], [ %25, %13 ]
  %16 = add nuw nsw i64 %15, %8
  %17 = getelementptr float, float* %0, i64 %16
  %18 = load float, float* %17, align 4
  %19 = shl nuw nsw i64 %15, 4
  %20 = add nuw nsw i64 %19, %10
  %21 = getelementptr float, float* %1, i64 %20
  %22 = load float, float* %21, align 4
  %23 = fmul float %18, %22
  %24 = fadd float %14, %23
  store float %24, float* %12, align 4
  %25 = add nuw nsw i64 %15, 1
  %26 = icmp ult i64 %15, 1432
  br i1 %26, label %13, label %27

27:                                               ; preds = %13
  %28 = add nuw nsw i64 %10, 1
  %29 = icmp ult i64 %10, 15
  br i1 %29, label %.preheader3, label %30

30:                                               ; preds = %27
  %31 = add nuw nsw i64 %7, 1
  %32 = icmp ult i64 %7, 2707
  br i1 %32, label %.preheader4, label %33

33:                                               ; preds = %30
  %34 = tail call dereferenceable_or_null(173376) i8* @malloc(i64 173376)
  %35 = ptrtoint i8* %34 to i64
  %36 = add i64 %35, 63
  %37 = and i64 %36, -64
  %38 = inttoptr i64 %37 to float*
  tail call void @llvm.memcpy.p0f32.p0f32.i64(float* noundef nonnull align 64 dereferenceable(173312) %38, float* noundef nonnull align 1 dereferenceable(173312) %3, i64 173312, i1 false)
  br label %.preheader2

.preheader2:                                      ; preds = %33, %62
  %39 = phi i64 [ 0, %33 ], [ %63, %62 ]
  %40 = mul nuw nsw i64 %39, 2708
  %41 = shl nuw nsw i64 %39, 4
  br label %.preheader

.preheader:                                       ; preds = %.preheader2, %59
  %42 = phi i64 [ 0, %.preheader2 ], [ %60, %59 ]
  %43 = add nuw nsw i64 %42, %41
  %44 = getelementptr float, float* %38, i64 %43
  %.pre5 = load float, float* %44, align 4
  br label %45

45:                                               ; preds = %.preheader, %45
  %46 = phi float [ %.pre5, %.preheader ], [ %56, %45 ]
  %47 = phi i64 [ 0, %.preheader ], [ %57, %45 ]
  %48 = add nuw nsw i64 %47, %40
  %49 = getelementptr float, float* %4, i64 %48
  %50 = load float, float* %49, align 4
  %51 = shl nuw nsw i64 %47, 4
  %52 = add nuw nsw i64 %51, %42
  %53 = getelementptr float, float* %2, i64 %52
  %54 = load float, float* %53, align 4
  %55 = fmul float %50, %54
  %56 = fadd float %46, %55
  store float %56, float* %44, align 4
  %57 = add nuw nsw i64 %47, 1
  %58 = icmp ult i64 %47, 2707
  br i1 %58, label %45, label %59

59:                                               ; preds = %45
  %60 = add nuw nsw i64 %42, 1
  %61 = icmp ult i64 %42, 15
  br i1 %61, label %.preheader, label %62

62:                                               ; preds = %59
  %63 = add nuw nsw i64 %39, 1
  %64 = icmp ult i64 %39, 2707
  br i1 %64, label %.preheader2, label %65

65:                                               ; preds = %62
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0f32.p0f32.i64(float* noalias nocapture writeonly, float* noalias nocapture readonly, i64, i1 immarg) #2

attributes #0 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #1 = { nofree nounwind }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
