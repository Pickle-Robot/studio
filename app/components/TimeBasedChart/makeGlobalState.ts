// This Source Code Form is subject to the terms of the Mozilla Public
// License, v2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/

import { EventEmitter } from "eventemitter3";
import { Dispatch, SetStateAction, useCallback, useEffect, useState } from "react";

// Create a "global" state hook
// Calling _makeGlobalState_ will return a hook function that behaves similar to useState
// The return value matches use state [value, setFunction].
//
// Any components which use this hook will receive updates to their value from any other components
// using the hook. The behavior is similar to if you wrapped each component using the hook under a
// context provider which shared the value and set function across the components.
//
// example:
// Initialize an instance at some global level (similar to createContext)
// const useGlobalState = makeGlobalState<string>("foobar");
//
// Then use within your component
// function Component() {
//   const [value, setValue] = useGlobalState({ enabled: true });
// }
//
// The enabled option (default true) is provided as a way to turn off global state updates
// If enabled is set to false, you will not receive updates to value
// You can still invoke updates on value
function makeGlobalState<T>() {
  const emitter = new EventEmitter();
  let existingValue: T | undefined;

  return function useGlobalState({
    enabled = true,
  }: {
    enabled?: boolean;
  }): [T | undefined, Dispatch<SetStateAction<T | undefined>>] {
    // if enabled, the user gets any existing value for the initial state
    // if we are not enabled then the user gets undefined for initial state
    const [localValue, setLocalValue] = useState<T | undefined>(
      enabled ? existingValue : undefined,
    );

    const setValue = useCallback((val: SetStateAction<T | undefined>) => {
      const newValue = val instanceof Function ? val(existingValue) : val;
      if (existingValue !== newValue) {
        existingValue = newValue;
        emitter.emit("change", existingValue);
      }
    }, []);

    useEffect(() => {
      if (!enabled) {
        return;
      }

      emitter.on("change", setLocalValue);
      return () => {
        // when unmounting, clear the global value to prevent stale entries
        // without this, the previous global value is retained which may no longer be relevant
        setValue(undefined);
        emitter.off("change", setLocalValue);
      };
    }, [enabled, setValue]);

    return [localValue, setValue];
  };
}

export default makeGlobalState;