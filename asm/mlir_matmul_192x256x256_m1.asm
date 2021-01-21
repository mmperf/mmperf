./matmul/matmul_192x256x256:
(__TEXT,__text) section
_rtclock:
0000000100003c68	sub	sp, sp, #0x30
0000000100003c6c	stp	x29, x30, [sp, #0x20]   ; Latency: 6
0000000100003c70	add	x29, sp, #0x20
0000000100003c74	add	x0, sp, #0x8
0000000100003c78	sub	x1, x29, #0x8
0000000100003c7c	bl	0x100003edc ; symbol stub for: _gettimeofday
0000000100003c80	cbz	w0, 0x100003c94
0000000100003c84	str	x0, [sp]                ; Latency: 4
0000000100003c88	adr	x0, #0x2e0 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003c8c	nop
0000000100003c90	bl	0x100003ee8 ; symbol stub for: _printf
0000000100003c94	ldr	d0, [sp, #0x8]          ; Latency: 4
0000000100003c98	scvtf	d0, d0                  ; Latency: 2
0000000100003c9c	ldr	s1, [sp, #0x10]         ; Latency: 4
0000000100003ca0	sshll.2d	v1, v1, #0x0    ; Latency: 2
0000000100003ca4	scvtf	d1, d1                  ; Latency: 2
0000000100003ca8	nop
0000000100003cac	ldr	d2, 0x100003f60         ; Latency: 4
0000000100003cb0	fmul	d1, d1, d2              ; Latency: 5
0000000100003cb4	fadd	d0, d1, d0              ; Latency: 5
0000000100003cb8	ldp	x29, x30, [sp, #0x20]   ; Latency: 4
0000000100003cbc	add	sp, sp, #0x30
0000000100003cc0	ret
_init_matrix:
0000000100003cc4	stp	x26, x25, [sp, #-0x50]! ; Latency: 6
0000000100003cc8	stp	x24, x23, [sp, #0x10]   ; Latency: 6
0000000100003ccc	stp	x22, x21, [sp, #0x20]   ; Latency: 6
0000000100003cd0	stp	x20, x19, [sp, #0x30]   ; Latency: 6
0000000100003cd4	stp	x29, x30, [sp, #0x40]   ; Latency: 6
0000000100003cd8	add	x29, sp, #0x40
0000000100003cdc	cmp	w2, #0x1
0000000100003ce0	b.lt	0x100003d38
0000000100003ce4	cmp	w1, #0x1
0000000100003ce8	b.lt	0x100003d38
0000000100003cec	mov	x19, x0                 ; Latency: 2
0000000100003cf0	mov	x20, #0x0
0000000100003cf4	mov	w21, w1                 ; Latency: 2
0000000100003cf8	mov	w22, w2                 ; Latency: 2
0000000100003cfc	lsl	x23, x21, #2
0000000100003d00	mov	w24, #0x30000000
0000000100003d04	mov	x25, x21                ; Latency: 2
0000000100003d08	mov	x26, x19                ; Latency: 2
0000000100003d0c	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003d10	scvtf	s0, w0                  ; Latency: 10
0000000100003d14	fmov	s1, w24                 ; Latency: 5
0000000100003d18	fmul	s0, s0, s1              ; Latency: 4
0000000100003d1c	str	s0, [x26], #0x4         ; Latency: 4
0000000100003d20	subs	x25, x25, #0x1
0000000100003d24	b.ne	0x100003d0c
0000000100003d28	add	x20, x20, #0x1
0000000100003d2c	add	x19, x19, x23           ; Latency: 2
0000000100003d30	cmp	x20, x22                ; Latency: 2
0000000100003d34	b.ne	0x100003d04
0000000100003d38	ldp	x29, x30, [sp, #0x40]   ; Latency: 4
0000000100003d3c	ldp	x20, x19, [sp, #0x30]   ; Latency: 4
0000000100003d40	ldp	x22, x21, [sp, #0x20]   ; Latency: 4
0000000100003d44	ldp	x24, x23, [sp, #0x10]   ; Latency: 4
0000000100003d48	ldp	x26, x25, [sp], #0x50   ; Latency: 4
0000000100003d4c	ret
_main:
0000000100003d50	sub	sp, sp, #0x60
0000000100003d54	stp	d9, d8, [sp, #0x20]     ; Latency: 6
0000000100003d58	stp	x22, x21, [sp, #0x30]   ; Latency: 6
0000000100003d5c	stp	x20, x19, [sp, #0x40]   ; Latency: 6
0000000100003d60	stp	x29, x30, [sp, #0x50]   ; Latency: 6
0000000100003d64	add	x29, sp, #0x50
0000000100003d68	add	x0, sp, #0x8
0000000100003d6c	add	x1, sp, #0x18
0000000100003d70	bl	0x100003edc ; symbol stub for: _gettimeofday
0000000100003d74	cbz	w0, 0x100003d88
0000000100003d78	str	x0, [sp]                ; Latency: 4
0000000100003d7c	adr	x0, #0x1ec ; literal pool for: "Error return from gettimeofday: %d"
0000000100003d80	nop
0000000100003d84	bl	0x100003ee8 ; symbol stub for: _printf
0000000100003d88	mov	x20, #0x0
0000000100003d8c	ldr	x19, [sp, #0x8]         ; Latency: 4
0000000100003d90	ldr	s0, [sp, #0x10]         ; Latency: 4
0000000100003d94	sshll.2d	v0, v0, #0x0    ; Latency: 2
0000000100003d98	scvtf	d0, d0                  ; Latency: 2
0000000100003d9c	nop
0000000100003da0	ldr	d8, 0x100003f60         ; Latency: 4
0000000100003da4	fmul	d9, d0, d8              ; Latency: 5
0000000100003da8	mov	w21, #0xc0
0000000100003dac	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003db0	subs	x21, x21, #0x1
0000000100003db4	b.ne	0x100003dac
0000000100003db8	add	x20, x20, #0x1
0000000100003dbc	cmp	x20, #0x100
0000000100003dc0	b.ne	0x100003da8
0000000100003dc4	mov	x20, #0x0
0000000100003dc8	scvtf	d0, x19                 ; Latency: 10
0000000100003dcc	fadd	d9, d9, d0              ; Latency: 5
0000000100003dd0	mov	w19, #0x100
0000000100003dd4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003dd8	subs	x19, x19, #0x1
0000000100003ddc	b.ne	0x100003dd4
0000000100003de0	add	x20, x20, #0x1
0000000100003de4	cmp	x20, #0x100
0000000100003de8	b.ne	0x100003dd0
0000000100003dec	mov	x19, #0x0
0000000100003df0	mov	w20, #0xc0
0000000100003df4	bl	0x100003ef4 ; symbol stub for: _rand
0000000100003df8	subs	x20, x20, #0x1
0000000100003dfc	b.ne	0x100003df4
0000000100003e00	add	x19, x19, #0x1
0000000100003e04	cmp	x19, #0x100
0000000100003e08	b.ne	0x100003df0
0000000100003e0c	add	x0, sp, #0x8
0000000100003e10	add	x1, sp, #0x18
0000000100003e14	bl	0x100003edc ; symbol stub for: _gettimeofday
0000000100003e18	cbz	w0, 0x100003e2c
0000000100003e1c	str	x0, [sp]                ; Latency: 4
0000000100003e20	adr	x0, #0x148 ; literal pool for: "Error return from gettimeofday: %d"
0000000100003e24	nop
0000000100003e28	bl	0x100003ee8 ; symbol stub for: _printf
0000000100003e2c	ldr	d0, [sp, #0x8]          ; Latency: 4
0000000100003e30	scvtf	d0, d0                  ; Latency: 2
0000000100003e34	ldr	s1, [sp, #0x10]         ; Latency: 4
0000000100003e38	sshll.2d	v1, v1, #0x0    ; Latency: 2
0000000100003e3c	scvtf	d1, d1                  ; Latency: 2
0000000100003e40	fmul	d1, d1, d8              ; Latency: 5
0000000100003e44	fadd	d8, d1, d0              ; Latency: 5
0000000100003e48	adr	x0, #0x143 ; literal pool for: "FILE_NAME"
0000000100003e4c	nop
0000000100003e50	adr	x1, #0x145 ; literal pool for: "w"
0000000100003e54	nop
0000000100003e58	bl	0x100003ec4 ; symbol stub for: _fopen
0000000100003e5c	mov	x19, x0                 ; Latency: 2
0000000100003e60	fsub	d0, d8, d9              ; Latency: 5
0000000100003e64	mov	x8, #0xc00000000000
0000000100003e68	movk	x8, #0x41e2, lsl #48
0000000100003e6c	fmov	d1, x8                  ; Latency: 5
0000000100003e70	fdiv	d0, d1, d0              ; Latency: 17
0000000100003e74	mov	x8, #0xcd6500000000
0000000100003e78	movk	x8, #0x41cd, lsl #48
0000000100003e7c	fmov	d1, x8                  ; Latency: 5
0000000100003e80	fdiv	d0, d0, d1              ; Latency: 17
0000000100003e84	str	d0, [sp]                ; Latency: 4
0000000100003e88	adr	x1, #0x10f ; literal pool for: "%0.2lf GFLOPS\n"
0000000100003e8c	nop
0000000100003e90	bl	0x100003ed0 ; symbol stub for: _fprintf
0000000100003e94	mov	x0, x19                 ; Latency: 2
0000000100003e98	bl	0x100003eb8 ; symbol stub for: _fclose
0000000100003e9c	mov	w0, #0x0
0000000100003ea0	ldp	x29, x30, [sp, #0x50]   ; Latency: 4
0000000100003ea4	ldp	x20, x19, [sp, #0x40]   ; Latency: 4
0000000100003ea8	ldp	x22, x21, [sp, #0x30]   ; Latency: 4
0000000100003eac	ldp	d9, d8, [sp, #0x20]     ; Latency: 4
0000000100003eb0	add	sp, sp, #0x60
0000000100003eb4	ret
