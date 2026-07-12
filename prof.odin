package main

import "base:runtime"
import "core:mem"
import "core:prof/spall"
import "core:sync"

Profiling_Mode_Type :: enum {
	None,
	Frame,
	All_Funcs, // Caution: This has lots of overhead!
	Custom, // Only capture custom begin/end events
}

PROF_MODE :: Profiling_Mode_Type(#config(PROF_MODE, 0))

when PROF_MODE != .None {
	spall_ctx: spall.Context

	@(thread_local)
	spall_buffer: spall.Buffer
	@(thread_local)
	buffer_backing: []u8
	@(thread_local)
	prof_allocator: mem.Allocator
}

when PROF_MODE == .All_Funcs {
	@(instrumentation_enter)
	spall_enter :: proc "contextless" (
		proc_address, call_site_return_address: rawptr,
		loc: runtime.Source_Code_Location,
	) {
		if spall_buffer.data == nil do return
		spall._buffer_begin(&spall_ctx, &spall_buffer, "", "", loc)
	}

	@(instrumentation_exit)
	spall_exit :: proc "contextless" (
		proc_address, call_site_return_address: rawptr,
		loc: runtime.Source_Code_Location,
	) {
		spall._buffer_end(&spall_ctx, &spall_buffer)
	}
}

// Call once at the very start of the main thread
prof_init :: proc(allocator := context.allocator) {
	when PROF_MODE != .None {
		spall_ctx = spall.context_create("trace.spall")
		prof_thread_init(allocator)
	}
}

// Call once per spawned thread to allocate the thread-local buffer
prof_thread_init :: proc(allocator := context.allocator) {
	when PROF_MODE != .None {
		prof_allocator = allocator
		buffer_backing = make([]u8, spall.BUFFER_DEFAULT_SIZE, allocator)
		spall_buffer = spall.buffer_create(buffer_backing, u32(sync.current_thread_id()))
	}
}

// Call once at the very end of the main thread
prof_deinit :: proc() {
	defer when PROF_MODE != .None {
		prof_thread_deinit()
		spall.context_destroy(&spall_ctx)
	}
}

// Call once at the very end of each spawned thread
prof_thread_deinit :: proc() {
	defer when PROF_MODE != .None {
		spall.buffer_destroy(&spall_ctx, &spall_buffer)
		delete(buffer_backing, prof_allocator)
	}
}

// Create a scoped event for each frame of your game/application which automatically
// ends itself at the end of the caller's scope.
prof_frame :: proc(label := "Frame", loc := #caller_location) {
	when PROF_MODE == .Frame {
		spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, label, "", loc)
	}
}

// Create a sub-event for the parts of each frame (ex. input, update, draw).
// Use within separate dedicate functions for each frame part for simplicity.
prof_frame_part :: proc(loc := #caller_location) {
	when PROF_MODE == .Frame {
		spall.SCOPED_EVENT(&spall_ctx, &spall_buffer, loc.procedure, "", loc)
	}
}

// A lower level profiling primitive to be called at the start of a section
prof_begin :: proc(label := "", loc := #caller_location) {
	when PROF_MODE != .None {
		spall._buffer_begin(&spall_ctx, &spall_buffer, label, "", loc)
	}
}

// A lower level profiling primitive to be called at the end of a section
prof_end :: proc(label := "", loc := #caller_location) {
	when PROF_MODE != .None {
		spall._buffer_end(&spall_ctx, &spall_buffer)
	}
}
